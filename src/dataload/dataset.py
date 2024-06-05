import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np


class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)
    
    
class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, abs_entity_neighbors, subcategory_neighbors):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors

        self.abs_entity_neighbors = abs_entity_neighbors
        self.subcategory_neighbors = subcategory_neighbors
        # for key in self.subcategory_neighbors:
        #     self.subcategory_neighbors[key] = torch.tensor(self.subcategory_neighbors[key], dtype=torch.int64)


    def line_mapper(self, line, sum_num_news):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # ------------------ News Subgraph ---------------------
        top_k = len(click_id)
        # 把新闻id转换为内部索引
        click_idx = self.trans_to_nindex(click_id)
        # 初始化源索引为点击新闻的索引
        source_idx = click_idx
        # 迭代 k_hops 次，从点击的新闻出发，扩展邻居新闻
        for _ in range(self.cfg.model.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                # 获取当前新闻的邻居新闻，数量为 num_neighbors
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            # 更新源索引为当前的邻居新闻
            source_idx = current_hop_idx
            # 将当前邻居新闻添加到点击新闻列表中
            click_idx.extend(current_hop_idx)

        # 构建子图，并获得子图中节点的映射索引
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        # 对映射索引进行填充，使其长度达到 his_size，填充值为 -1。mode:constant: 常量填充
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size-len(mapping_idx), 0), "constant", -1)

        
        # ------------------ Candidate News ---------------------
        label = 0
        # 把正样本和负样本转变为新闻索引。sess_pos + sess_neg: 列表相加，如[1, 2] + [3] = [1, 2, 3]
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        abs_candidate_entity = None
        abs_entity_mask = None
        subcategories = None

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            # 提取候选新闻的实体部分，大小为 [batch_size, entity_size]
            # TODO 维度改变
            origin_entity = candidate_input[:, -8 - self.cfg.model.entity_size:-8]  #[5, 5]
            # origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]  #[5, 5]
            # print(f"origin_entity: {origin_entity}")
            # 初始化候选实体邻居的矩阵，大小为 [(npratio+1) * entity_size, entity_neighbors]。 npratio: 负样本比例
            candidate_neighbor_entity = np.zeros(((self.cfg.npratio+1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64) # [5*5, 20]
            # 遍历所有实体索引
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]


            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio+1, self.cfg.model.entity_size *self.cfg.model.entity_neighbors) # [5, 5*20]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            # 合并原始实体和邻居实体
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
            # print(f"type(candidate_entity) = {type(candidate_entity)}")
            # print(f"candidate_entity: {candidate_entity}")

            # --------------Abstract Entity Graph-----------------
            if self.cfg.model.use_abs_entity:
                origin_abs_entity = candidate_input[:, -5:]
                # print(f"origin_abs_entity: {origin_abs_entity}")
                candidate_abs_neighbor_entity = np.zeros(((self.cfg.npratio+1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
                for cnt, idx in enumerate(origin_abs_entity.flatten()):
                    if idx == 0: continue
                    abs_entity_dict_length = len(self.abs_entity_neighbors[idx])
                    if abs_entity_dict_length == 0: continue
                    valid_len = min(abs_entity_dict_length, self.cfg.model.entity_neighbors)
                    candidate_abs_neighbor_entity[cnt, :valid_len] = self.abs_entity_neighbors[idx][:valid_len]
                candidate_abs_neighbor_entity = candidate_abs_neighbor_entity.reshape(self.cfg.npratio+1, self.cfg.model.entity_size *self.cfg.model.entity_neighbors)
                abs_entity_mask = candidate_abs_neighbor_entity.copy()
                entity_mask[abs_entity_mask > 0] = 1
                abs_candidate_entity = np.concatenate((origin_abs_entity, candidate_abs_neighbor_entity), axis=-1)
                # print(f"abs_candidate_entity: {abs_candidate_entity}")
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)


        # ------------------Subcategory Graph-------------------
        if self.cfg.model.use_subcategory_graph:
            origin_subcategory = candidate_input[:, -7:-6]
            candidate_neighbor_subcategory = np.zeros((self.cfg.npratio+1, self.cfg.model.subcategory_neighbors), dtype=np.int64)

            # print(f"origin_subcategory.shape: {origin_subcategory.shape}")
            # subcategory_dict_length = len(self.subcategory_neighbors)
            # valid_len = min(subcategory_dict_length, self.cfg.model.subcategory_neighbors)
            # print(f"origin_subcategory: {origin_subcategory}")
            # print(f"subcategory_neighbors: {self.subcategory_neighbors}")
            # candidate_neighbor_subcategory[:valid_len] = self.subcategory_neighbors[origin_subcategory][:valid_len]

            for cnt, sub_idx in enumerate(origin_subcategory):
                # print(f"cnt = {cnt}, sub_idx = {sub_idx}")
                subcategory_idx = sub_idx[0]
                # print(f"subcategory_idx = {subcategory_idx}")
                if subcategory_idx == 0: continue
                idx_tuple = (subcategory_idx,)
                # print(f"idx_tuple = {idx_tuple}")
                # print(f"idx_tuple[0] = {idx_tuple[0]}")
                if idx_tuple[0] not in self.subcategory_neighbors:
                    continue
                subcategory_dict_length = len(self.subcategory_neighbors[subcategory_idx])
                # print(f"subcategory_dict_length = {subcategory_dict_length}")
                # print(f"subcategory_dict_length: {subcategory_dict_length}")
                valid_len = min(subcategory_dict_length, self.cfg.model.subcategory_neighbors)
                # print(f"valid_len = {valid_len}")
                # print(f"candidate_neighbor_subcategory valid len: {valid_len}")
                candidate_neighbor_subcategory[cnt, :valid_len] = self.subcategory_neighbors[idx_tuple[0]][:valid_len]
                # print(f"candidate_neighbor_subcategory of subcategory idx{sub_idx}: {self.subcategory_neighbors[idx_tuple[0]][:valid_len]}")

            # TODO ?
            candidate_neighbor_subcategory.reshape(self.cfg.npratio+1, self.cfg.model.subcategory_neighbors)
            # print(f"type(origin_subcategory): {type(origin_subcategory)}")
            # print(f"type(candidate_neighbor_subcategory): {type(candidate_neighbor_subcategory)}")
            subcategories = np.concatenate((origin_subcategory, candidate_neighbor_subcategory), axis=-1)
            # print(f"type(subcategories) = {type(subcategories)}")
            # print(f"subcategories: {subcategories}")
            # subcategories = torch.tensor(subcategories, dtype=torch.int64)
            # print(f"type(subcategories) = {type(subcategories)}")
            # print(f"subcategories: {subcategories}")

        # return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
        #        sum_num_news+sub_news_graph.num_nodes
        # print(f"abs_candidate: {abs_candidate_entity}")
        # TODO 改变返回参数
        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
            sum_num_news + sub_news_graph.num_nodes, abs_candidate_entity, abs_entity_mask, subcategories

    def build_subgraph(self, subset, k, sum_num_nodes):
        # k: 选择的前k个结点
        # sum_num_nodes: 累积结点的基数，通常在处理多个结点时使用
        device = self.news_graph.x.device

        # subset: 要构建子图的结点子集
        if not subset: 
            subset = [0]
            
        subset = torch.tensor(subset, dtype=torch.long, device=device)
        
        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr, relabel_nodes=True, num_nodes=self.news_graph.num_nodes)
                    
        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k]+sum_num_nodes
    
    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            candidate_abs_entity_list = []
            abs_entity_mask_list = []
            candidate_subcategory_list = []

            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news, abs_candidate_entity, abs_entity_mask, subcategories = self.line_mapper(line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    candidate_abs_entity_list.append(torch.from_numpy(abs_candidate_entity))
                    abs_entity_mask_list.append(torch.from_numpy(abs_entity_mask))

                    # candidate_subcategory_list.append(subcategories)
                    # TODO subcategories是聚合的clicked_subcategory和candidate_subcategory
                    candidate_subcategory_list.append(torch.from_numpy(subcategories))

                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)
                        candidate_abs_entity_list = torch.stack(candidate_abs_entity_list)
                        abs_entity_mask_list = torch.stack(abs_entity_mask_list)
                        candidate_subcategory_list = torch.stack(candidate_subcategory_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels, candidate_abs_entity_list, abs_entity_mask_list, candidate_subcategory_list
                        clicked_graphs, mappings ,candidates, labels, candidate_entity_list, entity_mask_list, candidate_abs_entity_list, abs_entity_mask_list, candidate_subcategory_list  = [], [], [], [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)
                    candidate_abs_entity_list = torch.stack(candidate_abs_entity_list)
                    abs_entity_mask_list = torch.stack(abs_entity_mask_list)
                    candidate_subcategory_list = torch.stack(candidate_subcategory_list)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels, candidate_abs_entity_list, abs_entity_mask_list, candidate_subcategory_list
                    f.seek(0)


class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity, abs_entity_neighbors, news_abs_entity, subcategory_neighbors, news_subcategory):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, abs_entity_neighbors, subcategory_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.news_abs_entity = news_abs_entity
        self.news_subcategory = news_subcategory

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        # print(f"click_id = {click_id}")
        click_idx = self.trans_to_nindex(click_id)
        # print(f"click_idx = {click_idx}")
        clicked_entity = self.news_entity[click_idx]
        clicked_abs_entity = self.news_abs_entity[click_idx]
        clicked_subcategory = self.news_subcategory[click_idx]

        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

         # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]
        # print(f"candidate_index = {candidate_index}")
        # print(f"len(candidate_index) = {len(candidate_index)}")

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index)*self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]
            
            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size *self.cfg.model.entity_neighbors)
       
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        abs_candidate_entity, abs_entity_mask, subcategories = None, None, None
        # ------------------ Abstract Entity ---------------
        if self.cfg.model.use_abs_entity:
            origin_abs_entity = self.news_abs_entity[candidate_index]
            candidate_abs_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_abs_entity.flatten()):
                if idx == 0: continue
                abs_entity_dict_length = len(self.abs_entity_neighbors[idx])
                if abs_entity_dict_length == 0: continue
                valid_len = min(abs_entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_abs_neighbor_entity[cnt, :valid_len] = self.abs_entity_neighbors[idx][:valid_len]

            candidate_abs_neighbor_entity = candidate_abs_neighbor_entity.reshape(len(candidate_index),
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            abs_entity_mask = candidate_abs_neighbor_entity.copy()
            abs_entity_mask[abs_entity_mask > 0] = 1

            abs_candidate_entity = np.concatenate((origin_abs_entity, candidate_abs_neighbor_entity), axis=-1)
            # print(f"abs_candidate_entity: {abs_candidate_entity}")

        # ------------------- Subcategory Graph ------------------
        if self.cfg.model.use_subcategory_graph:
            origin_subcategory = self.news_subcategory[candidate_index]
            candidate_neighbor_subcategory = np.zeros((len(candidate_index), self.cfg.model.subcategory_neighbors),
                                                      dtype=np.int64)
            # print(f"origin_subcategory.shape: {origin_subcategory.shape}")
            for cnt, sub_idx in enumerate(origin_subcategory):
                # print(f"cnt = {cnt}, sub_idx = {sub_idx}")
                subcategory_idx = sub_idx[0]
                # print(f"subcategory_idx = {subcategory_idx}")
                if subcategory_idx == 0: continue
                idx_tuple = (subcategory_idx,)
                # print(f"idx_tuple = {idx_tuple}")
                # print(f"idx_tuple[0] = {idx_tuple[0]}")
                if idx_tuple[0] not in self.subcategory_neighbors:
                    continue
                subcategory_dict_length = len(self.subcategory_neighbors[subcategory_idx])
                # print(f"subcategory_dict_length = {subcategory_dict_length}")
                # print(f"subcategory_dict_length: {subcategory_dict_length}")
                valid_len = min(subcategory_dict_length, self.cfg.model.subcategory_neighbors)
                # print(f"valid_len = {valid_len}")
                # print(f"candidate_neighbor_subcategory valid len: {valid_len}")
                candidate_neighbor_subcategory[cnt, :valid_len] = self.subcategory_neighbors[idx_tuple[0]][:valid_len]
                # print(f"candidate_neighbor_subcategory of subcategory idx{sub_idx}: {self.subcategory_neighbors[idx_tuple[0]][:valid_len]}")

            # TODO ?
            candidate_neighbor_subcategory.reshape(len(candidate_index), self.cfg.model.subcategory_neighbors)
            # print(f"type(origin_subcategory): {type(origin_subcategory)}")
            # print(f"type(candidate_neighbor_subcategory): {type(candidate_neighbor_subcategory)}")
            subcategories = np.concatenate((origin_subcategory, candidate_neighbor_subcategory), axis=-1)
            # print(f"type(subcategories) = {type(subcategories)}")
            # print(f"subcategories: {subcategories}")


        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, clicked_abs_entity, abs_candidate_entity, abs_entity_mask, clicked_subcategory, subcategories
    
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, clicked_abs_entity, abs_candidate_entity, abs_entity_mask, clicked_subcategory, subcategories = self.line_mapper(line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, clicked_abs_entity, abs_candidate_entity, abs_entity_mask, clicked_subcategory, subcategories


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


