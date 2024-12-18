import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential

from src.models.base.layers import AttentionPooling


# class UserEncoder(nn.Module):
#     def __init__(self, cfg):
#
#         super().__init__()
#         self.news_dim = cfg.model.head_num * cfg.model.head_dim
#         # layers
#         self.atte = Sequential('x, mask', [
#             (MultiHeadAttention(self.news_dim,
#                                self.news_dim,
#                                self.news_dim,
#                                cfg.model.head_num,
#                                cfg.model.head_dim), 'x,x,x,mask -> x'),
#
#             (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x, mask -> x'),
#         ])
#
#     def forward(self, clicked_news, clicked_mask=None):
#         result = self.atte(clicked_news, clicked_mask)
#         return result

class UserTotalEncoder(nn.Module):
    def __init__(self, cfg):
        super(UserTotalEncoder, self).__init__()
        self.cfg = cfg
        # self.fc1 = nn.Linear(1200, 600)
        self.fc1 = nn.Linear(1200, 600)
        self.fc2 = nn.Linear(600, 400)
        self.relu = nn.LeakyReLU(0.2)
        self.news_dim = 400
        # self.atte = Sequential('a,b,c', [
        #     (lambda a, b, c: torch.stack([a, b, c], dim=1).view(-1, 3, self.news_dim), 'a,b,c -> x'),
        #     AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
        # ])


    def forward(self, user_common_emb, user_event_emb, hie_emb):
        user_emb = torch.cat((user_common_emb, user_event_emb, hie_emb), dim=-1)
        # print(f"user_emb.shape: {user_emb.shape}") # [32, 1200]

        # print(f"user_emb.shape: {user_emb.shape}")
        x = self.relu(self.fc1(user_emb))
        # print(f"x.shape: {x.shape}")
        result = self.fc2(x)
        # print(f"result.shape: {result.shape}")
        return result
        # print(f"user_total_encoder:")
        # print(f"user_common_emb.shape: {user_common_emb.shape}")
        # print(f"user_event_emb.shape: {user_event_emb.shape}")
        # print(f"hie_emb.shape: {hie_emb.shape}")
        # result = self.atte(user_common_emb.unsqueeze(0), user_event_emb.unsqueeze(0), hie_emb.unsqueeze(0))
        # print(f"result.shape: {result.shape}")

        # return result

    # def forward(self, user_common_emb, user_event_emb):
    #     user_emb = torch.cat((user_common_emb, user_event_emb), dim=-1)
    #     x = self.relu(self.fc1(user_emb))
    #     result = self.fc2(x)
    #     return result