import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential

from src.models.base.layers import AttentionPooling


# TODO Candidate Encoder
class CandidateEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_entity = cfg.model.use_entity
        self.use_abs_entity = cfg.model.use_abs_entity
        self.use_subcategory = cfg.model.use_subcategory_graph

        self.entity_dim = 100
        self.subcategory_dim = 100
        self.news_dim = cfg.model.head_dim * cfg.model.head_num
        self.output_dim = cfg.model.head_dim * cfg.model.head_num

        if self.use_entity:
            self.atte = Sequential('a,b,c', [
                (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        else:
            self.atte = Sequential('a,b,c', [
                (nn.Linear(self.news_dim, self.output_dim),'a -> x'),
                nn.LeakyReLU(0.2),
            ])

        if self.use_entity and self.use_abs_entity and self.use_subcategory:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,b,c,d,e,f,g], dim=-2).view(-1, 7, self.news_dim), 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        elif self.use_entity and self.use_abs_entity:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,b,c,d,e], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        elif self.use_entity and self.use_subcategory:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,b,c,f,g], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        elif self.use_abs_entity and self.use_subcategory:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,d,e,f,g], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        elif self.use_entity:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        elif self.use_abs_entity:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,d,e], dim=-2).view(-1, 3, self.news_dim),
                 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        elif self.use_subcategory:
            self.atte = Sequential('a,b,c,d,e,f,g', [
                (lambda a,b,c,d,e,f,g: torch.stack([a,f,g], dim=-2).view(-1, 3, self.news_dim),
                 'a,b,c,d,e,f,g -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])

    def forward(self, candidate_emb, origin_emb=None, neighbor_emb=None,
                cand_abs_origin_entity_emb=None, cand_abs_neighbor_entity_emb=None,
                cand_origin_subcategory_emb=None, cand_neighbor_subcategory_emb=None):

        batch_size, num_news = candidate_emb.shape[0], candidate_emb.shape[1]

        result = (self.atte(candidate_emb, origin_emb, neighbor_emb,
                           cand_abs_origin_entity_emb, cand_abs_neighbor_entity_emb,
                           cand_origin_subcategory_emb, cand_neighbor_subcategory_emb)
                  .view(batch_size, num_news, self.output_dim))

        return result