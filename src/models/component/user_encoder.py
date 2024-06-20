import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential

class UserEncoder(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        # layers
        self.atte = Sequential('x, mask', [
            (MultiHeadAttention(self.news_dim,
                               self.news_dim,
                               self.news_dim,
                               cfg.model.head_num,
                               cfg.model.head_dim), 'x,x,x,mask -> x'),

            (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x, mask -> x'),
        ])

    def forward(self, clicked_news, clicked_mask=None):
        result = self.atte(clicked_news, clicked_mask)
        return result

class UserTotalEncoder(nn.Module):
    def __init__(self, cfg):
        super(UserTotalEncoder, self).__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(800, 600)
        self.fc2 = nn.Linear(600, 400)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, user_common_emb, user_event_emb):
        user_emb = torch.cat((user_common_emb, user_event_emb), dim=-1)
        x = self.relu(self.fc1(user_emb))
        result = self.fc2(x)
        return result