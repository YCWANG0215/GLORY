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

from models.component.subcategory_encoder import SubcategoryEncoder

from src.models.base.layers import MultiHeadAttention, AttentionPooling
from src.models.component.HieNewsEncoder import HieNewsEncoder
from src.models.component.HieEncoder import HieEncoder
# from src.models.component.click_encoder import ClickTotalEncoder
from src.models.component.event_encoder import EventEncoder, EventAttentionEncoder, EventTotalEncoder
from src.models.component.gru_layer import GRULayer
from src.models.component.key_entity_encoder import KeyEntityEncoder, KeyEntityAttention
from src.models.component.user_encoder import UserTotalEncoder


# torch.autograd.set_detect_anomaly(True)

class GLORY(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None, abs_entity_emb=None, subcategory_dict=None, news_input=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity
        # self.use_abs_entity = cfg.model.use_abs_entity
        # self.use_subcategory = cfg.model.use_subcategory_graph
        self.use_event = cfg.model.use_event
        self.use_key_entity = cfg.model.use_key_entity
        # self.news_input = news_input
        # if self.use_subcategory:
        # self.subcategory_size = len(subcategory_dict)
        self.subcategory_size = cfg.dataset.subcategory_num
        self.category_size = cfg.dataset.category_num
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

        # TODO Event Encoder
        self.event_encoder = EventEncoder(cfg, glove_emb)
        self.event_attention_encoder = EventAttentionEncoder(cfg)
        self.event_total_encoder = EventTotalEncoder(cfg)


        # GCN
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),'x, index -> x'),
        ])

        # if self.use_key_entity:
        #     pretrain = torch.from_numpy(key_entity_emb)

        # Entity
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
            # self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
            # self.entity_encoder = EntityEncoder(cfg)
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

            self.atte = Sequential('x, mask', [
                (MultiHeadAttention(self.news_dim,
                                    self.news_dim,
                                    self.news_dim,
                                    cfg.model.head_num,
                                    cfg.model.head_dim), 'x,x,x,mask -> x'),

                (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim),
                 'x, mask -> x'),
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

        # if self.use_abs_entity:
        #     abs_pretrain = torch.from_numpy(abs_entity_emb).float()
        #     self.abs_entity_embedding_layer = nn.Embedding.from_pretrained(abs_pretrain, freeze=False, padding_idx=0)
        #
        #     self.local_abs_entity_encoder = Sequential('x, mask', [
        #         (self.abs_entity_embedding_layer, 'x -> x'),
        #         (EntityEncoder(cfg), 'x, mask -> x'),
        #     ])
        #
        #     self.global_abs_entity_encoder = Sequential('x, mask', [
        #         (self.abs_entity_embedding_layer, 'x -> x'),
        #         (GlobalEntityEncoder(cfg), 'x, mask -> x'),
        #     ])

        self.key_entity_attention = KeyEntityAttention(cfg)
        # if self.use_key_entity:
        #     key_pretrain = torch.from_numpy(entity_emb).float()
        #     self.key_entity_embedding_layer = nn.Embedding.from_pretrained(key_pretrain, freeze=False, padding_idx=0)
        #
        #     self.key_entity_encoder = Sequential('x, mask', [
        #         (self.key_entity_embedding_layer, 'x -> x'),
        #         (EntityEncoder(cfg), 'x, mask -> x'),
        #     ])
        # print(f"category_size: {self.category_size}")
        self.category_encoder = SubcategoryEncoder(self.category_size + 5)
        self.subcategory_encoder = SubcategoryEncoder(self.subcategory_size + 5)
        # if self.use_subcategory:
        #     self.subcategory_attention = SubcategoryAttention(cfg, self.subcategory_size)
        #     self.global_subcategory_encoder = GlobalSubcategoryEncoder(cfg)

        # HieRec
        if self.cfg.model.use_HieRec:
            self.hieRec_encoder = HieEncoder(cfg, HieNewsEncoder(cfg, glove_emb), entity_emb, self.category_encoder, self.subcategory_encoder)


        # self.event_gru = Sequential('x', [
        #     (nn.Linear(200, 300), 'x -> x'),
        #     (nn.Linear(300, 400), 'x -> x'),
        #     (nn.LeakyReLU(0.2), 'x -> x')
        # ])

        self.gru = GRULayer(cfg, 400, 400, self.cfg.model.gru_layer_num, 400)
        # self.key_entity_encoder = KeyEntityEncoder(cfg)

        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)
        # self.click_total_encoder = ClickTotalEncoder(cfg)
        # User Encoder
        # self.user_encoder = UserEncoder(cfg)
        self.user_total_encoder = UserTotalEncoder(cfg)
        # Candidate Encoder
        self.candidate_encoder = CandidateEncoder(cfg)

        # click prediction
        self.click_predictor = DotProduct()
        self.loss_fn = NCELoss()

    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, label=None, clicked_event=None, candidate_event=None, clicked_topic_list=None, clicked_topic_mask_list=None, clicked_subtopic_list=None, clicked_subtopic_mask_list=None, clicked_subtopic_news_list=None, clicked_subtopic_news_mask_list=None, clicked_event_mask=None, hetero_graph=None):
        # -------------------------------------- clicked ----------------------------------
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0
        # print(f"subgraph.x.shape = {subgraph.x.shape}")
        # print(f"[train] clicked_key_entity: {clicked_key_entity}")
        # print(f"######## clicked_key_entity.shape: {clicked_key_entity.shape}")
        # print(f"######## clicked_key_entity_mask.shape: {clicked_key_entity_mask.shape}")

        # print(f"candidate_news.shape: {candidate_news.shape}")
        # print(f"candidate_entity.shape: {candidate_entity.shape}")
        # print(f"candidate_subcategory.shape: {candidate_subcategory.shape}")

        # print(f"mapping_idx.shape: {mapping_idx.shape}") # [32, 50]    50 -> max click history size
        # print(f"mapping_idx: {mapping_idx}")
        # print(11)
        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        # TODO 维度
        clicked_entity = subgraph.x[mapping_idx, -13:-8]
        # print(f"clicked entity: {clicked_entity}")
        # clicked_entity = subgraph.x[mapping_idx, -8:-3]
        # clicked_abs_entity = subgraph.x[mapping_idx, -5:]
        # print(f"clicked abs_entity: {clicked_abs_entity}")
        # clicked_subcategory = subgraph.x[mapping_idx, -7:-6]
        # clicked_topic = subgraph.x[mapping_idx, -8:-7]

        # print(f"clicked subcategory: {clicked_subcategory}")
        # print(f"x.shape = {subgraph.x.shape}")  # torch.Size([15828, 43])
        # News Encoder + GCN
        x_flatten = subgraph.x.view(1, -1, token_dim)
        # print(f"x_flatten.shape = {x_flatten.shape}") # torch.Size([1, 15828, 43])
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)
        # print(f"x_encoded.shape = {x_encoded.shape}") # torch.Size([15828, 400])
        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)

        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)
        # print(f"clicked_origin_emb.shape = {clicked_origin_emb.shape}") # torch.Size([32, 50, 400])
        clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)

        # print(f"clicked_event.shape = {clicked_event.shape}")


        # print(12)
        # Attention pooling
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)
        else:
            clicked_entity = None
        # print("FINISH LINE 136...")
        # if self.use_abs_entity:
        #     clicked_abs_entity = self.local_abs_entity_encoder(clicked_abs_entity, None)
        # else:
        #     clicked_abs_entity = None
        # print("FINISH LINE 141...")
        # if self.use_subcategory:
        #     clicked_subcategory = self.subcategory_encoder(clicked_subcategory)
        # else:
        #     clicked_subcategory = None

        if self.use_event:
            clicked_event = self.event_encoder(clicked_event)
            # print(f"[train] clicked_event type = {type(clicked_event)}") # torch.Size([32, 400])
            clicked_event_emb = self.event_attention_encoder(clicked_event, clicked_event_mask)
            # print(f"clicked_event.shape: {clicked_event.shape}")
            # print(f"click_event_attention.shape: {clicked_event_attention.shape}")
            # TODO GRU
            # clicked_event = self.event_gru(clicked_event)
            # print(f"[train] clicked_event.shape: {clicked_event.shape}") # train: [32, 50, 400]
            # print(f"[train] clicked_event_mask: {clicked_event_mask}")
            # lengths = clicked_event_mask.sum(dim=1).long().cpu()
            # print(f"[train] lengths = {lengths}")
            # zero_length_indices = lengths == 0
            # print(f"[train] zero_length_indices.shape = {zero_length_indices.shape}") # [32]
            # 如果有长度为0的序列，填充一个全零向量，并将长度加1
            # if zero_length_indices.any():
                # zero_padding = torch.zeros(zero_length_indices.sum().item(), 1, 400) # [2, 1, 400]
                # zero_padding = torch.zeros(1, 400).half().cuda()
                # print(f"[train] zero_padding.shape = {zero_padding.shape}")
                # clicked_event[zero_length_indices] = zero_padding.squeeze(1)
                # print(f"clicked_event type: {clicked_event.dtype}")
                # print(f"zero_padding type: {zero_padding.dtype}")
                # clicked_event[zero_length_indices] = zero_padding
                # lengths[zero_length_indices] = 1
            # print(f"lengths.shape: {lengths.shape}") # [32]
            # print(f"[train] lengths = {lengths}")
            # clicked_event_gru = self.gru(clicked_event, lengths)
            # print(f"[train] clicked_event_gru type: {clicked_event_gru}")
            # print(f"[train] clicked_event_attention type: {clicked_event_attention}")
            # user_event_emb = self.event_total_encoder(clicked_event_attention, clicked_event_gru)
            # print(f"[train] clicked_event.shape: {clicked_event.shape}")
            # print(f"[train] clicked_event type: {type(clicked_event)}")
            # print(f"[train] clicked_event_gru type: {type(clicked_event_gru)}")
            # print(f"[train] clicked_origin_emb type: {type(clicked_origin_emb)}")
            # print(f"[train] clicked_event.shape: {clicked_event.shape}")
        else:
            clicked_event_emb = None
        # print(f"candidate_entity.dtype: {candidate_entity.dtype}")
        # print(f"candidate_entity_mask.dtype: {entity_mask.dtype}")
        # print(f"!!!!!!!!!!clicked_key_entity.shape: {clicked_key_entity.shape}")
        # clicked_key_entity_padding_news_mask = (clicked_key_entity_mask.sum(dim=2) != 0)
        # print(f"clicked_key_entity_padding_news_mask.shape: {clicked_key_entity_padding_news_mask.shape}")
        # if self.use_key_entity:
            # print(f"type(clicked_key_entity_mask): {type(clicked_key_entity_mask)}")
            # print(f"clicked_key_entity_mask: {clicked_key_entity_mask}")
            # clicked_key_entity_mask = clicked_key_entity != 0
            # clicked_key_entity_mask = torch.tensor(clicked_key_entity_mask, dtype=torch.long)
            # print(f"clicked_key_entity.dtype: {clicked_key_entity.dtype}")
            # print(f"clicked_key_entity_mask.dtype: {clicked_key_entity_mask.dtype}")
            # print(f"clicked_key_entity.shape: {clicked_key_entity.shape}")
            # print(f"clicked_key_entity_mask.shape: {clicked_key_entity_mask.shape}")
            # print(f"clicked_key_entity: {clicked_key_entity}")
            # print(f"[train] clicked_key_entity_emb_mask: {clicked_key_entity_padding_news_mask}")
            # clicked_key_entity_emb = self.key_entity_encoder(clicked_key_entity, clicked_key_entity_mask)
        # else:
        #     clicked_key_entity_emb = None

        # HieRec
        hie_emb = self.hieRec_encoder(clicked_topic_list, clicked_topic_mask_list,
                            clicked_subtopic_list, clicked_subtopic_mask_list, clicked_subtopic_news_list, clicked_subtopic_news_mask_list)


        # TODO clicked_total_embedding
        # clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
        # clicked_key_entity_final_emb = self.key_entity_attention(clicked_key_entity_emb, clicked_key_entity_padding_news_mask)
        # TODO 要带mapping_idx吗
        # print(f"clicked_origin_emb.shape: {clicked_origin_emb.shape}")
        # print(f"mask.shape: {mask.shape}")
        # print(f"clicked_graph.shape: {clicked_graph_emb.shape}")
        # print(f"mapping_idx.shape: {mapping_idx.shape}")
        # print(f"clicked_entity.shape: {clicked_entity.shape}")
        # print(f"entity_mask.shape: {entity_mask.shape}")
        # clicked_origin_emb = self.atte(clicked_origin_emb, None)
        # clicked_graph_emb = self.atte(clicked_graph_emb, None)
        # clicked_entity = self.atte(clicked_entity, None)
        # print(f"hie_emb.shape: {hie_emb.shape}") # [32, 400]
        # print(f"clicked_origin_emb.shape: {clicked_origin_emb.shape}") # [32, 50, 400]
        # print(f"clicked_graph_emb.shape: {clicked_graph_emb.shape}") # [32, 50, 400]
        # print(f"clicked_entity.shape: {clicked_entity.shape}") # [32, 50, 400]
        clicked_common_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
        # print(f"clicked_common_emb.shape: {clicked_common_emb.shape}") # [32, 50, 400]
        # clicked_total_emb = self.click_total_encoder(clicked_common_emb, clicked_event_emb)

        # print(f"clicked_total_emb.shape: {clicked_total_emb.shape}")

        # user_emb = self.user_encoder(clicked_total_emb, mask)
        # print(f"clicked_common_emb.shape: {clicked_common_emb.shape}")
        # user_common_emb = self.user_encoder(clicked_common_emb, None)
        # print(f"user_common_emb.shape: {user_common_emb.shape}")
        # print(f"user_event_emb.shape: {user_event_emb.shape}")
        # user_emb = self.user_total_encoder(user_common_emb, user_event_emb, clicked_key_entity_final_emb)
        # print(f"clicked_event.shape: {clicked_event.shape}")
        clicked_common_atte = self.atte(clicked_common_emb, None) # [32, 50]
        # print(f"clicked_event_emb.shape: {clicked_event_emb.shape}")
        # print(f"clicked_common_atte.shape: {clicked_common_atte.shape}")
        user_emb = self.user_total_encoder(clicked_common_atte, clicked_event_emb, hie_emb)
        # print(f"user_emb.shape: {user_emb.shape}") # [32, 400]

        # print(13)
        # ----------------------------------------- Candidate------------------------------------
        # print(f"candidate_news.shape: {candidate_news.shape}") # [32, 5, 43]
        cand_title_emb = self.local_news_encoder(candidate_news)                                      # [8, 5, 400]
        # print(f"cand_title_emb.shape: {cand_title_emb.shape}")
        if self.use_entity:
            # print(f"candidate_entity.shape: {candidate_entity.shape}")
            origin_entity, neighbor_entity = candidate_entity.split([self.cfg.model.entity_size,  self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)
            # print(f"origin_entity.shape: {origin_entity.shape}")
            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            # print(f"neighbor_entity.shape: {neighbor_entity.shape}") # [32, 5, 50]
            # print(f"entity_mask.shape: {entity_mask.shape}") # [32, 5, 50]
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)
            # print("FINISH LINE 159")
            # cand_entity_emb = self.entity_encoder(candidate_entity, entity_mask).view(batch_size, -1, self.news_dim) # [8, 5, 400]
        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None
        # print("FINISH LINE 163")
        # if self.use_abs_entity:
        #     abs_origin_entity, abs_neighbor_entity = candidate_abs_entity.split(
        #         [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)
        #
        #     cand_abs_origin_entity_emb = self.local_abs_entity_encoder(abs_origin_entity, None)
        #     cand_abs_neighbor_entity_emb = self.global_abs_entity_encoder(abs_neighbor_entity, abs_entity_mask)
        # else:
        #     cand_abs_origin_entity_emb, cand_abs_neighbor_entity_emb = None, None

        # if self.use_subcategory:
        #     # print(f"train: candidate_subcategory: {candidate_subcategory}")
        #     origin_subcategory, neighbor_subcategory = candidate_subcategory.split(
        #         [1, self.cfg.model.subcategory_neighbors], dim=-1)
        #     # print(f"train: origin_subcategory = {origin_subcategory}")
        #     # print(f"train: neighbor_subcategory = {neighbor_subcategory}")
        #     cand_origin_subcategory_emb = self.subcategory_encoder(origin_subcategory)
        #     neighbor_subcategory_emb = self.subcategory_attention(neighbor_subcategory)
        #     cand_neighbor_subcategory_emb = self.global_subcategory_encoder(neighbor_subcategory_emb)
        # else:
        #     cand_origin_subcategory_emb, cand_neighbor_subcategory_emb = None, None

        if self.use_event:
            cand_event_emb = self.event_encoder(candidate_event)
            # cand_event_emb = self.event_gru(cand_event_emb)
        else:
            cand_event_emb = None

        # if self.use_key_entity:
            # print(f"candidate_key_entity: {candidate_key_entity}")
            # print(f"candidate_key_entity.shape: {candidate_key_entity.shape}") # [32, 5, 8, 100]
            # print(f"candidate_key_entity_mask.shape: {candidate_key_entity_mask.shape}") # [32, 5, 8]
            # cand_key_entity_emb = self.key_entity_encoder(candidate_key_entity, candidate_key_entity_mask) # [32, 5, 400]
            # print(f"candidate_key_entity.shape: {candidate_key_entity.shape}")
            # print(f"candidate_key_entity_mask.shape: {candidate_key_entity_mask.shape}")
            # cand_key_entity_emb = self.key_entity_encoder(candidate_key_entity, candidate_key_entity_mask)
            # print(f"cand_key_entity_emb.shape: {cand_key_entity_emb.shape}")
        # else:
        #     cand_key_entity_emb = None

        # print("FINISH LINE 182")
        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb,
                                                cand_event_emb)
        # cand_final_emb = self.atte(cand_final_emb, None)
        # print(f"cand_final_emb.shape: {cand_final_emb.shape}")
        # print(f"user_emb.shape: {user_emb.shape}")
        # ----------------------------------------- Score ------------------------------------
        score = self.click_predictor(cand_final_emb, user_emb)
        loss = self.loss_fn(score, label)
        # print(14)
        return loss, score

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask, clicked_event, candidate_event, clicked_topic_list=None, clicked_topic_mask_list=None, clicked_subtopic_list=None, clicked_subtopic_mask_list=None, clicked_subtopic_news_list=None, clicked_subtopic_news_mask_list=None, clicked_event_mask=None, hetero_graph=None):
        
        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        #--------------------Attention Pooling
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
            # print(f"clicked_entity_emb.shape: {clicked_entity_emb.shape}")
        else:
            clicked_entity_emb = None

        # if self.use_abs_entity:
        #     clicked_abs_entity_emb = self.local_abs_entity_encoder(clicked_abs_entity.unsqueeze(0), None)
        # else:
        #     clicked_abs_entity_emb = None
        #
        # if self.use_subcategory:
        #     clicked_subcategory_emb = self.subcategory_encoder(clicked_subcategory.unsqueeze(0))
        # else:
        #     clicked_subcategory_emb = None

        if self.use_event:
            # num_news不一定有多少条
            # 数据格式：[1, num_news, 400]
            # print(f"[val] clicked_event.shape: {clicked_event.shape}")
            # clicked_event_emb = self.event_encoder(clicked_event.unsqueeze(0), None)
            valid_clicked_event = clicked_event[-num_news:, :].view(batch_size, num_news, news_dim)
            # print(f"[val] [batch_size, num_news, news_dim]: [{batch_size, num_news, news_dim}]")
            # print(f"[val] valid_clicked_event.shape: {valid_clicked_event.shape}")
            # print(f"clicked_event_emb.shape: {clicked_event_emb.shape}")
            # TODO GRU

            # clicked_event = valid_clicked_event
            # clicked_event = self.event_encoder(valid_clicked_event, clicked_event_mask)
            # print(f"click_event: {clicked_event}")
            # clicked_event = self.event_gru(clicked_event)
            # print(f"[train] clicked_event.shape: {clicked_event.shape}") # train: [32, 50, 400]
            # print(f"[train] clicked_event_mask: {clicked_event_mask}")
            # lengths = clicked_event_mask.sum(dim=1).long().cpu()
            # print(f"[train] lengths = {lengths}")
            # zero_length_indices = lengths == 0
            # print(f"[train] zero_length_indices.shape = {zero_length_indices.shape}") # [32]
            # 如果有长度为0的序列，填充一个全零向量，并将长度加1
            # if zero_length_indices.any():
                # zero_padding = torch.zeros(zero_length_indices.sum().item(), 1, 400) # [2, 1, 400]
                # zero_padding = torch.zeros(1, 400).half().cuda()
                # print(f"[train] zero_padding.shape = {zero_padding.shape}")
                # clicked_event[zero_length_indices] = zero_padding.squeeze(1)
                # print(f"clicked_event type: {clicked_event.dtype}")
                # print(f"zero_padding type: {zero_padding.dtype}")
                # clicked_event[zero_length_indices] = zero_padding
                # lengths[zero_length_indices] = 1
            # print(f"lengths.shape: {lengths.shape}") # [32]
            # print(f"[train] lengths = {lengths}")

            # print(f"clicked_event_mask.shape: {clicked_event_mask.shape}")
            # print(f"[val] clicked_event_mask: {clicked_event_mask}")
            # lengths = clicked_event_mask[-num_news:].sum().long().view(-1).cpu()
            # print(f"[val] lengths: {lengths}")
            # print(f"clicked_event.shape: {clicked_event.shape}")
            # print(f"valid_clicked_event.shape: {valid_clicked_event.shape}")
            clicked_event_emb = self.event_attention_encoder(valid_clicked_event, None)
            # clicked_event_emb = self.event_attention_encoder(clicked_event, clicked_event_mask)
            # print(f"clicked_event_attention_emb.shape: {clicked_event_attention_emb.shape}")
            # clicked_event_gru_emb = self.gru(valid_clicked_event, lengths)
            # user_event_emb = self.event_total_encoder(clicked_event_attention_emb)
            # print(f"user_event_emb.shape: {user_event_emb.shape}")

            # valid_clicked_emb = clicked_event_mask == True
            # clicked_event_emb = self.gru(clicked_event_emb, valid_clicked_emb)

        else:
            clicked_event_emb = None


        # print(f"[val] clicked_key_entity.shape: {clicked_key_entity.shape}") # [50, 8, 100]
        # print(f"[val] clicked_Key_entity_mask.shape: {clicked_key_entity_mask.shape}") # [50, 8]
        # clicked_key_entity_padding_news_mask = (clicked_key_entity_mask.sum(dim=1) != 0)
        # print(f"[train] clicked_key_entity_emb_mask: {clicked_key_entity_padding_news_mask}")
        # print(f"[val] clicked_key_entity_padding_news_mask.shape: {clicked_key_entity_padding_news_mask.shape}") # [50]
        # if self.use_key_entity:
        #     clicked_key_entity_emb = self.key_entity_encoder(clicked_key_entity.unsqueeze(0), clicked_key_entity_mask.unsqueeze(0))
        # else:
        #     clicked_key_entity_emb = None
        # print(f"clicked_key_entity.shape: {clicked_key_entity.shape}")
        # print(f"clicked_key_entity_emb.shape: {clicked_key_entity_emb.shape}")
        # print(f"clicked_origin_emb.shape: {clicked_origin_emb.shape}")
        # print(f"clicked_entity_emb.shape: {clicked_entity_emb.shape}")
        # print(f"[val] clicked_key_entity_emb.shape: {clicked_key_entity_emb.shape}")
        # clicked_key_entity_final_emb = self.key_entity_attention(clicked_key_entity_emb, clicked_key_entity_padding_news_mask.unsqueeze(0))
        # print(f"len(clicked_subtopic_news_list): {len(clicked_subtopic_news_list)}")
        # stack_clicked_subtopic_news = []
        # clicked_subtopic_news_list = [torch.tensor(sublist) if not isinstance(sublist, torch.Tensor) else sublist
        #                               for sublist in clicked_subtopic_news_list]
        # clicked_subtopic_news_list = torch.stack(clicked_subtopic_news_list)
        # 检查 clicked_subtopic_news_list 中的每个元素是否是张量

        # 将列表合并成一个张量
        # clicked_subtopic_news_list_tensor = torch.stack(clicked_subtopic_news_list)

        # clicked_subtopic_news = torch.cat(clicked_subtopic_news_list, dim=0).cpu().numpy()
        # print(f"clicked_subtopic_news.shape: {clicked_subtopic_news.shape}")
        clicked_hie_emb = self.hieRec_encoder(clicked_topic_list, clicked_topic_mask_list,
                            clicked_subtopic_list, clicked_subtopic_mask_list, clicked_subtopic_news_list, clicked_subtopic_news_mask_list)
        clicked_common_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)
        # print(f"clicked_common_emb.shape: {clicked_common_emb.shape} ")
        # print(f"[val] clicked_common_emb: {clicked_common_emb}")
        clicked_common_atte = self.atte(clicked_common_emb, None)
        # TODO clicked_final_emb + clicked_event_emb
        # clicked_final_emb = self.click_total_encoder(clicked_common_emb, clicked_event_emb)

        # user_emb = self.user_encoder(clicked_final_emb)  # [1, 400]
        # user_common_emb = self.user_encoder(clicked_common_emb)
        # print(f"clicked_common_atte.shape: {clicked_common_atte.shape}")
        # print(f"clicked_event_emb.shape: {clicked_event_emb.shape}")
        # print(f"clicked_hie_emb.shape: {clicked_hie_emb.shape}")
        user_emb = self.user_total_encoder(clicked_common_atte, clicked_event_emb, clicked_hie_emb)


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

        # if self.use_abs_entity:
        #     cand_abs_entity_input = candidate_abs_entity.unsqueeze(0)
        #     abs_entity_mask = cand_abs_entity_mask.unsqueeze(0)
        #     origin_abs_entity, abs_neighbor_entity = cand_abs_entity_input.split([self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)
        #
        #     cand_origin_abs_entity_emb = self.local_abs_entity_encoder(origin_abs_entity, None)
        #     cand_neighbor_abs_entity_emb = self.global_abs_entity_encoder(abs_neighbor_entity, abs_entity_mask)
        #
        # else:
        #     cand_origin_abs_entity_emb, cand_neighbor_abs_entity_emb = None, None

        # if self.use_subcategory:
        #     cand_subcategory_input = candidate_subcategory.unsqueeze(0)
            # print(f"val: cand_subcategory_input: {cand_subcategory_input}")
            # origin_subcategory, neighbor_subcategory = cand_subcategory_input.split([1, self.cfg.model.subcategory_neighbors], dim=-1)
            # print(f"val: origin_subcategory: {origin_subcategory}")
            # print(f"val: neighbor_subcategory: {neighbor_subcategory}")
            # cand_origin_subcategory_emb = self.subcategory_encoder(origin_subcategory)
            # print(f"val: neighbor_subcategory.shape: {neighbor_subcategory.shape}")
            # neighbor_subcategory_emb = self.subcategory_attention(neighbor_subcategory)
            # print(f"val: neighbor_subcategory_emb.shape: {neighbor_subcategory_emb.shape}")
            # cand_neighbor_subcategory_emb = self.global_subcategory_encoder(neighbor_subcategory_emb)
            # print(f"val: cand_neighbor_subcategory_emb.shape: {cand_neighbor_subcategory_emb.shape}")
        # else:
        #     cand_origin_subcategory_emb, cand_neighbor_subcategory_emb = None, None

        # if self.use_event:
        #     cand_event_emb = self.event_encoder(candidate_event, None)
        # else:
        #     cand_event_emb = None
        # if self.use_key_entity:
        #     cand_key_entity_emb = self.key_entity_encoder(candidate_key_entity.unsqueeze(0), candidate_key_entity_mask.unsqueeze(0))
        # else:
        #     cand_key_entity_emb = None

        cand_event_emb = candidate_event.unsqueeze(0)

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb, cand_neighbor_entity_emb,
                                                cand_event_emb)
        # ---------------------------------------------------------------------------------------
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores




















