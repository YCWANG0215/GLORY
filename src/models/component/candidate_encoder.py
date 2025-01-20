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
        # self.use_abs_entity = cfg.model.use_abs_entity
        # self.use_subcategory = cfg.model.use_subcategory_graph
        self.use_event = cfg.model.use_event
        self.use_key_entity = cfg.model.use_key_entity
        self.entity_dim = 100
        self.subcategory_dim = 100
        self.news_dim = cfg.model.head_dim * cfg.model.head_num
        self.output_dim = cfg.model.head_dim * cfg.model.head_num

        # if self.use_entity:
        #     self.atte = Sequential('a,b,c', [
        #         (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # else:
        #     self.atte = Sequential('a,b,c', [
        #         (nn.Linear(self.news_dim, self.output_dim),'a -> x'),
        #         nn.LeakyReLU(0.2),
        #     ])
        # if self.use_entity == False and self.use_abs_entity and self.use_subcategory and self.use_event and self.use_key_entity:
        #     # print(f"[candidate_encoder]: use_entity = False")
        #     self.atte = Sequential('a,b,c,d,e,f,g,h,i', [
        #         (lambda a, b, c, d, e, f, g, h, i: torch.stack([a, d, e, f, g, h, i], dim=-2).view(-1, 7,
        #                                                                                            self.news_dim),
        #          'a,b,c,d,e,f,g,h,i -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_entity and self.use_abs_entity and self.use_subcategory == False and self.use_event and self.use_key_entity:
        #     # no subcategory
        #     # print("[candidateEncoder]: use_subcategory = False")
        #     self.atte = Sequential('a,b,c,d,e,f,g,h,i', [
        #         (lambda a, b, c, d, e, f, g, h, i: torch.stack([a, b, c, d, e, h, i], dim=-2).view(-1, 7,
        #                                                                                                  self.news_dim),
        #          'a,b,c,d,e,f,g,h,i -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # if self.use_entity and self.use_abs_entity and self.use_subcategory and self.use_event and self.use_key_entity:
        #     # entity + abs_entity + subcategory_graph + event + key_entity
        #     self.atte = Sequential('a,b,c,d,e,f,g,h,i', [
        #         (lambda a, b, c, d, e, f, g, h,i: torch.stack([a, b, c, d, e, f, g, h, i], dim=-2).view(-1, 9, self.news_dim),
        #         'a,b,c,d,e,f,g,h,i -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_entity and self.use_abs_entity and self.use_subcategory and self.use_event and self.use_key_entity == False:
        #     # print(f"candidateEncoder: use_key_entity = False")
        #     self.atte = Sequential('a,b,c,d,e,f,g,h,i', [
        #         (lambda a, b, c, d, e, f, g, h, i: torch.stack([a, b, c, d, e, f, g, h], dim=-2).view(-1, 8, self.news_dim),
        #          'a,b,c,d,e,f,g,h,i -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_entity and self.use_abs_entity and self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,b,c,d,e,f,g], dim=-2).view(-1, 7, self.news_dim), 'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_entity and self.use_abs_entity:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,b,c,d,e], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_entity and self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,b,c,f,g], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_abs_entity and self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,d,e,f,g], dim=-2).view(-1, 5, self.news_dim), 'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_entity:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_abs_entity:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,d,e], dim=-2).view(-1, 3, self.news_dim),
        #          'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # elif self.use_subcategory:
        #     self.atte = Sequential('a,b,c,d,e,f,g', [
        #         (lambda a,b,c,d,e,f,g: torch.stack([a,f,g], dim=-2).view(-1, 3, self.news_dim),
        #          'a,b,c,d,e,f,g -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])
        # self.atte = Sequential('a,b,c,d,e', [
        #         (lambda a,b,c,d,e: torch.stack([a,b,c,d,e], dim=-2).view(-1, 5, self.news_dim),
        #          'a,b,c,d,e -> x'),
        #         AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        #         nn.Linear(self.news_dim, self.output_dim),
        #         nn.LeakyReLU(0.2),
        #     ])

        self.atte = Sequential('a,b,c,d', [
            (lambda a, b, c, d: torch.stack([a, b, c, d], dim=-2).view(-1, 4, self.news_dim),
             'a,b,c,d -> x'),
            AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            nn.Linear(self.news_dim, self.output_dim),
            nn.LeakyReLU(0.2),
        ])



    def forward(self, candidate_emb, origin_emb=None, neighbor_emb=None,
                cand_event_emb=None):

        batch_size, num_news = candidate_emb.shape[0], candidate_emb.shape[1]
        # print(f"batch_size: {batch_size}, num_news: {num_news}")
        # print(f"candidate emb shape: {candidate_emb.shape}")
        # print(f"origin emb shape: {origin_emb.shape}")
        # print(f"neighbor_emb shape: {neighbor_emb.shape}")
        # print(f"cand_event_emb shape: {cand_event_emb.shape}")
        # print(f"cand_key_entity_emb shape: {cand_key_entity_emb.shape}")
        # print(f"indirect_entity_neighbors_emb.shape: {indirect_entity_neighbors_emb.shape}")

        result = (self.atte(candidate_emb, origin_emb, neighbor_emb,
                           cand_event_emb)
                  .view(batch_size, num_news, self.output_dim))
        # result = (self.atte(candidate_emb, origin_emb, neighbor_emb,
        #                     cand_event_emb)
        #           .view(batch_size, num_news, self.output_dim))
        # print(f"result shape: {result.shape}")
        return result