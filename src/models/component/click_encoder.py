import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential

from src.models.base.layers import AttentionPooling


# TODO 改：最终候选新闻信息的聚合
class ClickEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = 400
        self.use_entity = cfg.model.use_entity
        self.use_abs_entity = cfg.model.use_abs_entity
        self.use_subcategory = cfg.model.use_subcategory_graph
        self.use_event = cfg.model.use_event
        self.use_key_entity = cfg.model.use_key_entity
        # print(f"ClickEncoder: use_entity={self.use_entity}, use_abs_entity={self.use_abs_entity}, use_subcategory_graph={self.use_subcategory}, use_event={self.use_event}")
        # if self.use_entity:
        #     self.atte = Sequential('a,b,c', [
        #         (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # else:
        #     self.atte = Sequential('a,b', [
        #         (lambda a,b: torch.stack([a,b], dim=-2).view(-1, 2, self.news_dim), 'a,b -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # if self.use_entity and self.use_abs_entity and self.use_subcategory and self.use_event:
        #     self.atte = Sequential('a,b,c,d,e,f', [
        #         (lambda a,b,c,d,e,f: torch.stack([a,b,c,d,e,f], dim=-2).view(-1, 6, self.news_dim), 'a,b,c,d,e,f -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim)
        #     ])
        # if self.use_entity and self.use_abs_entity and self.use_subcategory and self.use_key_entity:
        #     self.atte = Sequential('a,b,c,d,e,f', [
        #         (lambda a, b, c, d, e, f: torch.stack([a, b, c, d, e, f], dim=-2).view(-1, 6, self.news_dim),
        #          'a,b,c,d,e,f -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # if self.use_entity == False and self.use_abs_entity and self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d', [
        #         (lambda a, b, c, d: torch.stack([a, b, c, d], dim=-2).view(-1, 4, self.news_dim),
        #          'a,b,c,d -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # elif self.use_entity and self.use_abs_entity and self.use_subcategory == False:
        #     # print("[clickEncoder]: use_subcategory = False")
        #     self.atte = Sequential('a,b,c,d', [
        #         (lambda a, b, c, d: torch.stack([a, b, c, d], dim=-2).view(-1, 4, self.news_dim),
        #          'a,b,c,d -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])


        # elif self.use_entity and self.use_abs_entity:
        #     self.atte = Sequential('a,b,c,d', [
        #         (lambda a, b, c, d: torch.stack([a,b,c,d], dim=-2).view(-1, 4, self.news_dim),
        #          'a,b,c,d -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # elif self.use_entity and self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d', [
        #         (lambda a, b, c, d: torch.stack([a,b,c,d], dim=-2).view(-1, 4, self.news_dim),
        #          'a,b,c,d -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # elif self.use_abs_entity and self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d', [
        #         (lambda a, b, c, d, e: torch.stack([a,b,d,e], dim=-2).view(-1, 4, self.news_dim),
        #          'a,b,c,d -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # elif self.use_entity:
        #     self.atte = Sequential('a,b,c', [
        #         (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # elif self.use_abs_entity:
        #     self.atte = Sequential('a,b,c', [
        #         (lambda a,b,c,d,e: torch.stack([a,b,d], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # elif self.use_subcategory:
        #     self.atte = Sequential('a,b,c', [
        #         (lambda a,b,c,d,e: torch.stack([a,b,e], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # else:
        #     self.atte = Sequential('a,b', [
        #         (lambda a,b: torch.stack([a,b], dim=-2).view(-1, 2, self.news_dim), 'a,b -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #     ])
        # self.atte = Sequential('a,b,c', [
        #     (lambda a, b, c: torch.stack([a, b, c], dim=1).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #     AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        # ])
        self.atte = Sequential('a,b,c,d', [
            (lambda a, b, c, d: torch.stack([a, b, c, d], dim=1).view(-1, 4, self.news_dim), 'a,b,c,d -> x'),
            AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        ])



    def forward(self, click_title_emb, click_graph_emb, click_entity_emb=None, clicked_event_emb=None):

        batch_size, num_news = click_title_emb.shape[0], click_title_emb.shape[1]
        # print("in click_encoder: ")
        # print(f"clicked_title.shape: {click_title_emb.shape}")
        # print(f"clicked_graph.shape: {click_graph_emb.shape}")
        # print(f"clicked_entity.shape: {click_entity_emb.shape}")
        # print(f"clicked_key_entity.shape: {click_key_entity_emb.shape}")

        # print("over")
        # print(f"click_encoder: batch_size={batch_size}, num_news={num_news}")
        # if click_entity_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb)
        # else:
        #     result = self.atte(click_title_emb, click_graph_emb)
        # if click_entity_emb is not None and click_abs_entity_emb is not None and click_subcategory_emb is not None and click_event_emb is not None:
        #     # print("exe1")
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb, click_subcategory_emb, click_event_emb)
        # if click_entity_emb is not None and click_abs_entity_emb is not None and click_subcategory_emb is not None and clicked_key_entity_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb, click_subcategory_emb, clicked_key_entity_emb)
        # if click_entity_emb is None and click_abs_entity_emb is not None and click_subcategory_emb is not None:
        #     # print(f"[clickEncoder]: use_entity = false")
        #     result = self.atte(click_title_emb, click_graph_emb, click_abs_entity_emb, click_subcategory_emb)
        # elif click_entity_emb is not None and click_abs_entity_emb is not None and click_subcategory_emb is None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb)

        # if click_entity_emb is not None and click_abs_entity_emb is not None and click_subcategory_emb is not None:
        #     # print("exe2")
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb, click_subcategory_emb)
        # elif click_entity_emb is not None and click_abs_entity_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb)
        # elif click_entity_emb is not None and click_subcategory_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_subcategory_emb)
        # elif click_abs_entity_emb is not None and click_subcategory_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_abs_entity_emb, click_subcategory_emb)
        # elif click_entity_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb)
        # elif click_abs_entity_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_abs_entity_emb)
        # elif click_subcategory_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_subcategory_emb)
        # else:
        #     result = self.atte(click_title_emb, click_graph_emb)

        result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, clicked_event_emb)

        # return result.view(batch_size, num_news, self.news_dim)
        return result.view(batch_size, -1, self.news_dim)


# class ClickTotalEncoder(nn.Module):
#     def __init__(self, cfg):
#         super(ClickTotalEncoder, self).__init__()
#         self.cfg = cfg
#         self.fc1 = nn.Linear(800, 600)
#         self.fc2 = nn.Linear(600, 400)
#         self.relu = nn.LeakyReLU(0.2)
#
#     def forward(self, clicked_common_emb, clicked_event_emb):
#         # batch_size, num_news= event_encoder_input.shape
#         clicked_emb = torch.cat((clicked_common_emb, clicked_event_emb), dim=-1)
#         x = self.relu(self.fc1(clicked_emb))
#         result = self.fc2(x)
#         # print(f"event total encode result.shape: {result.shape}")
#         return result