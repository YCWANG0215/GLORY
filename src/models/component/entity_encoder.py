import torch
import torch.nn as nn
from models.base.layers import *
from torch_geometric.nn import Sequential

from src.models.base.layers import MultiHeadAttention, AttentionPooling


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = 400

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, int(self.entity_dim / cfg.model.head_dim), cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(self.entity_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Linear(self.entity_dim, self.news_dim),
            nn.LeakyReLU(0.2),
        ])



    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity, _ = entity_input.shape
        # print(f"entity input shape: {entity_input.shape}")
        # print(f"entity_input.dtype: {entity_input.dtype}")
        # entity_input = entity_input.float()
        if entity_mask is not None:
            # print(f"entity_mask.dtype: {entity_mask.dtype}")
            # entity_mask = entity_mask.float()
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask.view(batch_size*num_news, num_entity)).view(batch_size, num_news, self.news_dim)
        else:
            # print(f"entity_input.shape: {entity_input.shape}")
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), None).view(batch_size, num_news, self.news_dim)
        # print(f"entity encoder result shape: {result.shape}")
        return result

class GlobalEntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, cfg.model.head_num, cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
        ])


    def forward(self, entity_input, entity_mask=None):
        batch_size, num_news, num_entity,_ = entity_input.shape
        # print(f"batch_size: {batch_size}, num_news: {num_news}, num_entity: {num_entity}")
        if entity_mask is not None:
            entity_mask = entity_mask.view(batch_size*num_news, num_entity)
        # print(f"entity_input.shape: {entity_input.shape}")
        # print(f"entity_mask.shape: {entity_mask.shape}")
        result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask).view(batch_size, num_news, self.news_dim)
        # print(f"result.shape: {result.shape}")
        return result