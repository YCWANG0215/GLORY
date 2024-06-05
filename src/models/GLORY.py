import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv

from models.base.layers import *
from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *
from models.component.subcategory_encoder import *

from models.component.subcategory_encoder import SubcategoryEncoder, GlobalSubcategoryEncoder, SubcategoryAttention

# torch.autograd.set_detect_anomaly(True)

class GLORY(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None, abs_entity_emb=None, subcategory_dict=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity
        self.use_abs_entity = cfg.model.use_abs_entity
        self.use_subcategory = cfg.model.use_subcategory_graph
        self.subcategory_size = len(subcategory_dict)
        # print(f"subcategory size = {self.subcategory_size}")

        # TODO head_num->注意力头数量 head_dim->每个注意力头的维度
        # news_dim为什么定义为head_num * head_dim ?
        # news_dim计算多头注意力的输出维度
        # 多头注意力机制的输出维度是所有注意力头的输出连接结果，因此其总维度是head_num * head_dim
        self.news_dim =  cfg.model.head_num * cfg.model.head_dim

        # entity_dim = 100
        self.entity_dim = cfg.model.entity_emb_dim

        # -------------------------- Model --------------------------
        # News Encoder
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)


        # GCN
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),'x, index -> x'),
        ])

        # Entity
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
            # self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

            self.local_entity_encoder = Sequential('x, mask', [
                (entity_embedding_layer, 'x -> x'),
                # (self.entity_embedding_layer, 'x -> x'),
                (EntityEncoder(cfg), 'x, mask -> x'),
            ])

            self.global_entity_encoder = Sequential('x, mask', [
                (entity_embedding_layer, 'x -> x'),
                # (self.entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),
            ])

            # # TODO -------- 新增Start
            # self.entity_encoder = EntityEncoder(cfg, entity_emb)
            # self.news_encoder = Sequential('x, mask', [
            #     (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            #     (AttentionPooling(self.news_dim,
            #                       cfg.model.attention_hidden_dim), 'x,mask -> x'),
            #     nn.LayerNorm(self.news_dim),
            #     # nn.Linear(self.news_dim, self.news_dim),
            #     # nn.LeakyReLU(0.2),
            # ])
            # # TODO -------- 新增End

        if self.use_abs_entity:
            pretrain = torch.from_numpy(abs_entity_emb).float()
            self.abs_entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

            self.local_entity_encoder = Sequential('x, mask', [
                (self.abs_entity_embedding_layer, 'x -> x'),
                (EntityEncoder(cfg), 'x, mask -> x'),
            ])

            self.global_entity_encoder = Sequential('x, mask', [
                (self.abs_entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),
            ])

        if self.use_subcategory:
            self.subcategory_encoder = SubcategoryEncoder(self.subcategory_size)
            self.subcategory_attention = SubcategoryAttention(cfg, self.subcategory_size)
            self.global_subcategory_encoder = GlobalSubcategoryEncoder(cfg)

        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)

        # User Encoder
        self.user_encoder = UserEncoder(cfg)
        
        # Candidate Encoder
        self.candidate_encoder = CandidateEncoder(cfg)

        # click prediction
        self.click_predictor = DotProduct()
        self.loss_fn = NCELoss()


    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, label=None, candidate_abs_entity=None, abs_entity_mask=None, candidate_subcategory=None,):
        # -------------------------------------- clicked ----------------------------------
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0

        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        # TODO 维度
        clicked_entity = subgraph.x[mapping_idx, -13:-8]
        # print(f"clicked entity: {clicked_entity}")
        # clicked_entity = subgraph.x[mapping_idx, -8:-3]
        clicked_abs_entity = subgraph.x[mapping_idx, -5:]
        # print(f"clicked abs_entity: {clicked_abs_entity}")
        clicked_subcategory = subgraph.x[mapping_idx, -7:-6]
        # print(f"clicked subcategory: {clicked_subcategory}")

        # News Encoder + GCN
        x_flatten = subgraph.x.view(1, -1, token_dim)
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)

        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)

        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)
        clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)

        # Attention pooling
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)
        else:
            clicked_entity = None
        # print("FINISH LINE 136...")
        if self.use_abs_entity:
            clicked_abs_entity = self.local_entity_encoder(clicked_abs_entity, None)
        else:
            clicked_abs_entity = None
        # print("FINISH LINE 141...")
        if self.use_subcategory:
            clicked_subcategory = self.subcategory_encoder(clicked_subcategory)
        else:
            clicked_subcategory = None

        # TODO clicked_total_embedding
        # clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
        clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity, clicked_abs_entity, clicked_subcategory)
        user_emb = self.user_encoder(clicked_total_emb, mask)

        # ----------------------------------------- Candidate------------------------------------
        cand_title_emb = self.local_news_encoder(candidate_news)                                      # [8, 5, 400]
        if self.use_entity:
            origin_entity, neighbor_entity = candidate_entity.split([self.cfg.model.entity_size,  self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)
            # print("FINISH LINE 159")
            # cand_entity_emb = self.entity_encoder(candidate_entity, entity_mask).view(batch_size, -1, self.news_dim) # [8, 5, 400]
        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None
        # print("FINISH LINE 163")
        if self.use_abs_entity:
            abs_origin_entity, abs_neighbor_entity = candidate_abs_entity.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_abs_origin_entity_emb = self.local_entity_encoder(abs_origin_entity, None)
            cand_abs_neighbor_entity_emb = self.global_entity_encoder(abs_neighbor_entity, abs_entity_mask)
        else:
            cand_abs_origin_entity_emb, cand_abs_neighbor_entity_emb = None, None

        if self.use_subcategory:
            # print(f"train: candidate_subcategory: {candidate_subcategory}")
            origin_subcategory, neighbor_subcategory = candidate_subcategory.split(
                [1, self.cfg.model.subcategory_neighbors], dim=-1)
            # print(f"train: origin_subcategory = {origin_subcategory}")
            # print(f"train: neighbor_subcategory = {neighbor_subcategory}")
            cand_origin_subcategory_emb = self.subcategory_encoder(origin_subcategory)
            neighbor_subcategory_emb = self.subcategory_attention(neighbor_subcategory)
            cand_neighbor_subcategory_emb = self.global_subcategory_encoder(neighbor_subcategory_emb)
        else:
            cand_origin_subcategory_emb, cand_neighbor_subcategory_emb = None, None

        # print("FINISH LINE 182")
        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb,
                                                cand_abs_origin_entity_emb, cand_abs_neighbor_entity_emb,
                                                cand_origin_subcategory_emb, cand_neighbor_subcategory_emb)

        # ----------------------------------------- Score ------------------------------------
        score = self.click_predictor(cand_final_emb, user_emb)
        loss = self.loss_fn(score, label)

        return loss, score

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask, clicked_abs_entity, candidate_abs_entity, cand_abs_entity_mask, clicked_subcategory, candidate_subcategory):
        
        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        #--------------------Attention Pooling
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
        else:
            clicked_entity_emb = None

        if self.use_abs_entity:
            clicked_abs_entity_emb = self.local_entity_encoder(clicked_abs_entity.unsqueeze(0), None)
        else:
            clicked_abs_entity_emb = None

        if self.use_subcategory:
            clicked_subcategory_emb = self.subcategory_encoder(clicked_subcategory.unsqueeze(0))
        else:
            clicked_subcategory_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb, clicked_abs_entity_emb, clicked_subcategory_emb)

        user_emb = self.user_encoder(clicked_final_emb)  # [1, 400]

        # ----------------------------------------- Candidate------------------------------------

        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            origin_entity, neighbor_entity = cand_entity_input.split([self.cfg.model.entity_size,  self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        if self.use_abs_entity:
            cand_abs_entity_input = candidate_abs_entity.unsqueeze(0)
            abs_entity_mask = cand_abs_entity_mask.unsqueeze(0)
            origin_abs_entity, abs_neighbor_entity = cand_abs_entity_input.split([self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_abs_entity_emb = self.local_entity_encoder(origin_abs_entity, None)
            cand_neighbor_abs_entity_emb = self.global_entity_encoder(abs_neighbor_entity, abs_entity_mask)

        else:
            cand_origin_abs_entity_emb, cand_neighbor_abs_entity_emb = None, None

        if self.use_subcategory:
            cand_subcategory_input = candidate_subcategory.unsqueeze(0)
            # print(f"val: cand_subcategory_input: {cand_subcategory_input}")
            origin_subcategory, neighbor_subcategory = cand_subcategory_input.split([1, self.cfg.model.subcategory_neighbors], dim=-1)
            # print(f"val: origin_subcategory: {origin_subcategory}")
            # print(f"val: neighbor_subcategory: {neighbor_subcategory}")
            cand_origin_subcategory_emb = self.subcategory_encoder(origin_subcategory)
            # print(f"val: neighbor_subcategory.shape: {neighbor_subcategory.shape}")
            neighbor_subcategory_emb = self.subcategory_attention(neighbor_subcategory)
            # print(f"val: neighbor_subcategory_emb.shape: {neighbor_subcategory_emb.shape}")
            cand_neighbor_subcategory_emb = self.global_subcategory_encoder(neighbor_subcategory_emb)
            # print(f"val: cand_neighbor_subcategory_emb.shape: {cand_neighbor_subcategory_emb.shape}")
        else:
            cand_origin_subcategory_emb, cand_neighbor_subcategory_emb = None, None

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb, cand_neighbor_entity_emb,
                                                cand_origin_abs_entity_emb, cand_neighbor_abs_entity_emb,
                                                cand_origin_subcategory_emb, cand_neighbor_subcategory_emb)
        # ---------------------------------------------------------------------------------------
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores




















