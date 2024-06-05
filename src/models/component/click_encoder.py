import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential

# TODO 改：最终候选新闻信息的聚合
class ClickEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = 400
        self.use_entity = cfg.model.use_entity
        self.use_abs_entity = cfg.model.use_abs_entity
        self.use_subcategory = cfg.model.use_subcategory_graph

        print(f"ClickEncoder: use_entity={self.use_entity}, use_abs_entity={self.use_abs_entity}, use_subcategory_graph={self.use_subcategory}")
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

        if self.use_entity and self.use_abs_entity and self.use_subcategory:
            self.atte = Sequential('a,b,c,d,e', [
                (lambda a,b,c,d,e: torch.stack([a,b,c,d,e], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        elif self.use_entity and self.use_abs_entity:
            self.atte = Sequential('a,b,c,d', [
                (lambda a, b, c, d: torch.stack([a,b,c,d], dim=-2).view(-1, 4, self.news_dim),
                 'a,b,c,d -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        elif self.use_entity and self.use_subcategory:
            self.atte = Sequential('a,b,c,d', [
                (lambda a, b, c, d: torch.stack([a,b,c,d], dim=-2).view(-1, 4, self.news_dim),
                 'a,b,c,d -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        elif self.use_abs_entity and self.use_subcategory:
            self.atte = Sequential('a,b,c,d', [
                (lambda a, b, c, d, e: torch.stack([a,b,d,e], dim=-2).view(-1, 4, self.news_dim),
                 'a,b,c,d -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        elif self.use_entity:
            self.atte = Sequential('a,b,c', [
                (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        elif self.use_abs_entity:
            self.atte = Sequential('a,b,c', [
                (lambda a,b,c,d,e: torch.stack([a,b,d], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        elif self.use_subcategory:
            self.atte = Sequential('a,b,c', [
                (lambda a,b,c,d,e: torch.stack([a,b,e], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        else:
            self.atte = Sequential('a,b', [
                (lambda a,b: torch.stack([a,b], dim=-2).view(-1, 2, self.news_dim), 'a,b -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])

    
    def forward(self, click_title_emb, click_graph_emb, click_entity_emb=None, click_abs_entity_emb=None, click_subcategory_emb=None):

        batch_size, num_news = click_title_emb.shape[0], click_title_emb.shape[1]
        # if click_entity_emb is not None:
        #     result = self.atte(click_title_emb, click_graph_emb, click_entity_emb)
        # else:
        #     result = self.atte(click_title_emb, click_graph_emb)

        if click_entity_emb is not None and click_abs_entity_emb is not None and click_subcategory_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb, click_subcategory_emb)
        elif click_entity_emb is not None and click_abs_entity_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_abs_entity_emb)
        elif click_entity_emb is not None and click_subcategory_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_entity_emb, click_subcategory_emb)
        elif click_abs_entity_emb is not None and click_subcategory_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_abs_entity_emb, click_subcategory_emb)
        elif click_entity_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_entity_emb)
        elif click_abs_entity_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_abs_entity_emb)
        elif click_subcategory_emb is not None:
            result = self.atte(click_title_emb, click_graph_emb, click_subcategory_emb)
        else:
            result = self.atte(click_title_emb, click_graph_emb)

        return result.view(batch_size, num_news, self.news_dim)
    
