import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle

from dataload.dataset import *

from utils.common import load_pretrain_emb

from utils.common import load_key_entity_emb

from dataload.dataset import EventDataset, NewsDataset


def load_data(cfg, mode='train', model=None, local_rank=0):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    # ------------- load news.tsv-------------
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    # print(f"len(news_dict): {len(news_index)}")
    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    # print(f"news_input.shape: {news_input.shape}")
    event_index = pickle.load(open(Path(data_dir[mode]) / "event_dict.bin", "rb"))
    # print(f"len(event_dict) = {len(event_index)}")
    event_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_event.bin", "rb"))
    event_type_dict = pickle.load(open(Path(data_dir[mode]) / "event_type_dict.bin", "rb"))
    # print(f"event_type_dict.len: {len(event_type_dict)}")
    # print(f"event_input.shape = {event_input.shape}")
    # key_entity_dict = pickle.load(open(Path(data_dir[mode]) / "key_entity_dict.bin", "rb"))
    key_entities = pickle.load(open(Path(data_dir[mode]) / "key_entities.bin", "rb"))
    # category_dict: 新闻类别字符串（如体育）到类别ID的映射
    topic_dict = pickle.load(open(Path(data_dir[mode]) / "category_dict.bin", "rb"))
    subtopic_dict = pickle.load(open(Path(data_dir[mode]) / "subcategory_dict.bin", "rb"))
    # news_subtopic_map: 新闻id到其子类别id的映射
    news_subtopic_map = pickle.load(open(Path(data_dir[mode]) / "news_subtopic_map.bin", "rb"))
    news_topic_map = pickle.load(open(Path(data_dir[mode]) / "news_topic_map.bin", "rb"))
    # user_history_map: 用户ID到其浏览记录的topic_group和subtopic_group的映射
    user_history_map = pickle.load(open(Path(data_dir[mode]) / "user_history_map.bin", "rb"))
    node_dict = json.load(open(Path(data_dir[mode]) / "node_dict.json", "rb"))
    node_index = json.load(open(Path(data_dir[mode]) / "node_index.json", "rb"))
    hetero_graph = torch.load(open(Path(data_dir[mode]) / "hetero_graph.pt", "rb"))
    hetero_graph_news_input = pickle.load(open(Path(data_dir[mode]) / "hetero_graph_news_input.bin", "rb"))

    # print(f"len(key_entities) = {len(key_entities)}")
    # key_entity_input_mask = pickle.load(open(Path(data_dir[mode]) / "key_entities_mask.bin", "rb"))

    entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"

    # key_entity_emb = None
    # config = cfg
    # print(f"news_input.shape = {news_input.shape}")
    # ------------- load behaviors_np{X}.tsv --------------
    if mode == 'train':
        key_entity_input, key_entity_input_mask = load_key_entity_emb(cfg, mode, 100, key_entities, news_index)
        # print(f"[Data_load-load_data]: key_entity_input: {key_entity_input}")
        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        # target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv"
        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")


            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            # if cfg.model.use_event:
            #     event_graph = torch.load(Path(data_dir[mode]) / "nltk_event_graph.pt")
            #     if cfg.model.directed is False:
            #         event_graph.edge_index, event_graph.edge_attr = to_undirected(event_graph.edge_index, event_graph.edge_attr)
            #     print(f"[{mode}] Event Graph Info: {event_graph}")

            if cfg.model.use_entity:
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            # 添加摘要实体图、类别图
            # if cfg.model.use_abs_entity:
            #     abs_entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "abs_entity_neighbor_dict.bin", "rb"))
            #     total_length = sum(len(lst) for lst in abs_entity_neighbors.values())
            #     print(f"[{mode}] abs_entity_neighbor list Length: {total_length}")
            # else:
            #     abs_entity_neighbors = None

            # if cfg.model.use_subcategory_graph:
            #     subcategory_neighbors = pickle.load(open(Path(data_dir[mode]) / "subcategory_neighbor_dict.bin", "rb"))
            #     # print(f"subcategory_neighbors: {subcategory_neighbors}")
            #     total_length = sum(len(lst) for lst in subcategory_neighbors.values())
            #     print(f"[{mode}] subcategory_neighbor list Length: {total_length}")
            # else:
            #     subcategory_neighbors = None





            # dataset = TrainGraphDataset(
            #     filename=target_file,
            #     news_index=news_index,
            #     news_input=news_input,
            #     local_rank=local_rank,
            #     cfg=cfg,
            #     neighbor_dict=news_neighbors_dict,
            #     news_graph=news_graph,
            #     entity_neighbors=entity_neighbors
            # )
            # TODO dataset()改动
            dataset = TrainGraphDataset(
                filename=target_file,
                news_index=news_index, # news_id(N开头) -> news_index(序号)
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors,
                # abs_entity_neighbors=abs_entity_neighbors,
                # subcategory_neighbors=subcategory_neighbors,
                event_index=event_index,
                event_input=event_input,
                # key_entity_index=key_entity_dict,
                # key_entity_input=key_entity_input,
                # key_entity_input_mask=key_entity_input_mask,
                topic_dict=topic_dict,
                subtopic_dict=subtopic_dict,
                news_topic_map=news_topic_map,
                news_subtopic_map=news_subtopic_map,
                user_history_map=user_history_map,
                node_dict=node_dict,
                node_index=node_index,
                hetero_graph=hetero_graph,
                hetero_graph_news_input=hetero_graph_news_input
            )
            dataloader = DataLoader(dataset, batch_size=None)
            
        else:
            dataset = TrainDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                event_index=event_index,
                event_input=event_input
            )

            dataloader = DataLoader(dataset,
                                    batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    # batch_size=int(cfg.batch_size / 1),
                                    pin_memory=True)
        return dataloader
    elif mode in ['val', 'test']:
        # convert the news to embeddings
        # key_entity_input, key_entity_input_mask = load_key_entity_emb(cfg, mode, 100, key_entities, news_index)
        news_dataset = NewsDataset(news_input)
        news_dataloader = DataLoader(news_dataset,
                                     batch_size=int(cfg.batch_size * cfg.gpu_num),
                                     # batch_size=int(cfg.batch_size / 1),
                                     num_workers=cfg.num_workers)
        # print(f"EventDataset event_input shape: {event_input.shape}")
        event_dataset = EventDataset(event_input)
        # print(f"event_dataset.shape: {event_dataset.shape}")
        event_dataloader = DataLoader(event_dataset,
                                      batch_size=int(cfg.batch_size * cfg.gpu_num),
                                      # batch_size=int(cfg.batch_size / 1),
                                      num_workers=cfg.num_workers)

        # hie_dataset = HieDataset(user_history_map)
        # hie_dataloader = DataLoader(hie_dataset,
        #                             batch_size=int(cfg.batch_size * cfg.gpu_num),
        #                             num_workers=cfg.num_workers)

        stacked_news = []
        with torch.no_grad():
            for news_batch in tqdm(news_dataloader, desc=f"[{local_rank}] Processing validation News Embedding"):
                # print(f"news_batch.shape: {news_batch.shape}") # [32, 43]
                # print(f"news_batch: {news_batch}")
                if cfg.model.use_graph:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                else:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                    # batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                # print(f"batch_emb.shape: {batch_emb.shape}") # [32, 400]
                stacked_news.append(batch_emb)
        # print(f"stacked_news shape: {len(stacked_news)}") # 2039
        news_emb = torch.cat(stacked_news, dim=0).cpu().numpy()
        # print(f"news_emb.shape: {news_emb.shape}") # [65239, 400]

        stacked_events = []
        with torch.no_grad():
            # local_rank = 'cpu'
            # idx = 1
            for event_batch in tqdm(event_dataloader, desc=f"[{local_rank}] Processing validation Event Embedding"):
                # print(f"event_batch.shape: {event_batch.shape}") # [32, 11]
                # print(f"event_batch: {event_batch}")
                # event_emb = model.module.event_encoder(event_batch.long().unsqueeze(0).to(local_rank), None).squeeze(0).detach()
                # TODO detach()
                # print(f"event idx{idx} join.")
                # event_emb = event_batch.long().unsqueeze(0).to(local_rank)
                # event_emb = event_batch.to(local_rank)
                # print(f"finish 1")
                # event_emb = model.module.event_encoder(event_emb, None)
                # print("finish 2")
                # event_emb = event_emb.squeeze(0)
                # print(f"finish 3")
                event_emb = model.module.event_encoder(event_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                # print(f"event idx{idx} event_emb finish.")
                # idx += 1
                # event_emb = model.module.event_encoder(event_batch.long().to(local_rank), None)
                # print(f"event_emb.shape: {event_emb.shape}") # [32, 400]
                # print(f"event_emb: {event_emb}")
                stacked_events.append(event_emb)
                # print(f"stacked_events: {stacked_events}")

        # print(f"stackd_events.shape: {stacked_events.shape}")
        events_emb = torch.cat(stacked_events, dim=0).cpu().numpy()
        # print(f"event_emb.shape: {event_emb.shape}")
        # print(f"events_emb: {events_emb}")

        # stacked_hie = []
        # with torch.no_grad():
        #     for hie_batch in tqdm(hie_dataloader, desc=f"[{local_rank}] Processing validation Hie Embedding"):
        #         cur_user_info = user_history_map
        #         topic_group = cur_user_info[0]
        #         subtopic_group = cur_user_info[1]
        #         sorted_topic_group = sorted(topic_group.items(), key=lambda item: len(item[1]), reverse=True)
        #         sorted_subtopic_group = sorted(subtopic_group.items(), key=lambda item: len(item[1]), reverse=True)
        #         topic_ids = [topic_id for topic_id, _ in sorted_topic_group]
        #         subtopic_ids = [subtopic_id for subtopic_id, _ in sorted_subtopic_group]
        #         subtopic_news_id_lists_tmp = [news_ids.copy() for _, news_ids in sorted_subtopic_group]
        #         subtopic_news_id_lists, subtopic_news_id_lists_mask = pad_or_truncate(subtopic_news_id_lists_tmp,
        #                                                                                    cfg.model.news_per_subtopic,
        #                                                                                    cfg.model.subtopic_per_user,
        #                                                                                    news_index,
        #                                                                                    0)
        #         topic_ids, topic_ids_mask = select_top_k_with_padding(topic_ids, cfg.model.topic_per_user)
        #         subtopic_ids, subtopic_ids_mask = select_top_k_with_padding(subtopic_ids, cfg.model.subtopic_per_user)
        #         subtopic_news_lists = []
        #         for i in range(len(subtopic_news_id_lists_mask)):
        #             cur_list = []
        #             for j in range(len(subtopic_news_id_lists_mask[i])):
        #                 if subtopic_news_id_lists_mask[i][j] != 0:
        #                     cur_list.append(news_input[subtopic_news_id_lists[i][j]])
        #                 else:
        #                     cur_list.append(np.array(np.zeros(43), dtype=np.int32))
        #             subtopic_news_lists.append(cur_list)
        #         hie_emb = model.module.hie_encoder(hie_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
        #
        #         stacked_hie.append(hie_emb)
        # hie_emb = torch.cat(stacked_hie, dim=0).cpu().numpy()
        # print(f"hie_emb.shape: {hie_emb.shape}")


        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))


            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            # if cfg.model.use_event:
            #     # event_graph = torch.load(Path(data_dir[mode]) / "nltk_event_graph.pt")
            #     # event_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "event_neighbor_dict.bin", "rb"))
            #     if cfg.model.directed is False:
            #         event_graph.edge_index, event_graph.edge_attr = to_undirected(event_graph.edge_index,
            #                                                                     event_graph.edge_attr)
            #     print(f"[{mode}] News Graph Info: {event_graph}")
            if cfg.model.use_entity:
                # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            # if cfg.model.use_HieRec:


            # if cfg.model.use_abs_entity:
            #     abs_entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "abs_entity_neighbor_dict.bin", "rb"))
            #     total_length = sum(len(lst) for lst in abs_entity_neighbors.values())
            #     print(f"[{mode}] abs_entity_neighbor list Length: {total_length}")
            # else:
            #     abs_entity_neighbors = None

            # if cfg.model.use_subcategory_graph:
            #     subcategory_neighbors = pickle.load(open(Path(data_dir[mode]) / "subcategory_neighbor_dict.bin", "rb"))
            #     total_length = sum(len(lst) for lst in subcategory_neighbors.values())
            #     print(f"[{mode}] subcategory_neighbor list Length: {total_length}")
            # else:
            #     subcategory_neighbors = None

            if mode == 'val':
                dataset = ValidGraphDataset(
                    # filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
                    filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    # TODO change
                    news_entity=news_input[:,-13:-8],
                    # news_entity=news_input[:,-8:-3],
                    entity_neighbors=entity_neighbors,
                    # abs_entity_neighbors=abs_entity_neighbors,
                    # news_abs_entity=news_input[:, -5:],
                    # subcategory_neighbors=subcategory_neighbors,
                    # news_subcategory=news_input[:, -7:-6],
                    event_index=event_index,
                    event_input=events_emb,
                    # key_entity_index=key_entity_dict,
                    # key_entity_input=key_entity_input,
                    # key_entity_input_mask=key_entity_input_mask,
                    topic_dict=topic_dict,
                    subtopic_dict=subtopic_dict,
                    news_topic_map = news_topic_map,
                    news_subtopic_map = news_subtopic_map,
                    user_history_map = user_history_map,
                    news_idx_input=news_input,
                    # hetero_graph=hetero_graph
                    node_dict=node_dict,
                    node_index=node_index,
                    hetero_graph=hetero_graph,
                    hetero_graph_news_input=hetero_graph_news_input
                )

            elif mode == 'test':
                dataset = ValidGraphDataset(
                    # filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    # TODO change
                    news_entity=news_input[:, -13:-8],
                    # news_entity=news_input[:,-8:-3],
                    entity_neighbors=entity_neighbors,
                    # abs_entity_neighbors=abs_entity_neighbors,
                    # news_abs_entity=news_input[:, -5:],
                    # subcategory_neighbors=subcategory_neighbors,
                    # news_subcategory=news_input[:, -7:-6],
                    event_index=event_index,
                    event_input=events_emb,
                    # key_entity_index=key_entity_dict,
                    # key_entity_input=key_entity_input,
                    # key_entity_input_mask=key_entity_input_mask,
                    topic_dict=topic_dict,
                    subtopic_dict=subtopic_dict,
                    news_topic_map=news_topic_map,
                    news_subtopic_map=news_subtopic_map,
                    user_history_map=user_history_map,
                    node_dict=node_dict,
                    node_index=node_index,
                    hetero_graph=hetero_graph,
                    hetero_graph_news_input=hetero_graph_news_input
                )

            dataloader = DataLoader(dataset, batch_size=None)
            # dataloader = DataLoader(dataset, batch_size=None, num_workers=cfg.num_workers)

        else:
            if mode == 'val':
                dataset = ValidDataset(
                    # filename=Path(data_dir[mode]) / f"behaviors_0.tsv",
                    filename=Path(data_dir[mode]) / f"behaviors_{local_rank}.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    event_index=event_index,
                    events_emb=events_emb,
                )
            else:
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    event_index=event_index,
                    events_emb=events_emb,
                )

            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    # batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    # pin_memory=True, # collate_fn already puts data to GPU
                                    # num_workers=cfg.num_workers,
                                    collate_fn=lambda b: collate_fn(b, local_rank))
        return dataloader


def collate_fn(tuple_list, local_rank):
    clicked_news = [x[0] for x in tuple_list]
    clicked_mask = [x[1] for x in tuple_list]
    candidate_news = [x[2] for x in tuple_list]
    clicked_index = [x[3] for x in tuple_list]
    candidate_index = [x[4] for x in tuple_list]

    if len(tuple_list[0]) == 6:
        labels = [x[5] for x in tuple_list]
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index, labels
    else:
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index

def pad_or_truncate(lst, k, k2, news_index, pad_value=0):
    result = []
    result_mask = []
    # print(f"input lst = {lst}")
    for sublist in lst:
        for i in range(len(sublist)):
            sublist[i] = news_index[sublist[i]]

    for sublist in lst:
        if len(sublist) < k:
            result.append(sublist + [pad_value] * (k - len(sublist)))
            mask_sublist = [1] * len(sublist) + [0] * (k - len(sublist))

            # result.append(news_input[sublist])
            # for i in range(k - len(sublist)):
            #     result.append([0] * 43)
            # mask_sublist = [1] * len(sublist) + [0] * (k - len(sublist))
        else:
            result.append(sublist[:k])
            mask_sublist = [1] * k
            # result.append(news_input[sublist[:k]])
            # mask_sublist = [1] * k
        result_mask.append(mask_sublist)

    if len(result) < k2:
        for i in range(k2 - len(result)):
            result.append([pad_value] * k)
            result_mask.append([0] * k)
            # tmp = []
            # for j in range(k):
            #     tmp.append([0] * 43)
            # result.append(tmp)
            # result_mask.append([0] * k)
    else:
        result = result[:k2]
        result_mask = result_mask[:k2]

    return result, result_mask


def select_top_k_with_padding(lst, k, padding_value=0):
    result = lst + [padding_value] * (k - len(lst)) if len(lst) < k else lst[:k]
    result_mask = [1] * len(lst) + [0] * (k - len(lst)) if len(lst) < k else [1] * k

    return result, result_mask