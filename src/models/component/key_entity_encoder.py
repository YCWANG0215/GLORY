import torch
import torch.nn as nn
from models.base.layers import *
from torch_geometric.nn import Sequential

from src.models.base.layers import MultiHeadAttention, AttentionPooling


class KeyEntityEncoder(nn.Module):
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

        batch_size, num_news, num_entity = entity_input.shape[0], entity_input.shape[1], entity_input.shape[2]
        # print(f"entity input shape: {entity_input.shape}")
        # print(f"entity_input.dtype: {entity_input.dtype}")
        entity_input = entity_input.float()
        if entity_mask is not None:
            # print(f"entity_mask.dtype: {entity_mask.dtype}")
            entity_mask = entity_mask.float()
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask.view(batch_size*num_news, num_entity)).view(batch_size, num_news, self.news_dim)
        else:
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), None).view(batch_size, num_news, self.news_dim)
        # print(f"entity encoder result shape: {result.shape}")
        return result



class KeyEntityAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb_dim = 400
        self.news_dim = 400
        self.attention = Sequential('x, mask', [
            # dropout专门用于训练，推理阶段要关掉dropout
            # dropout：在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态（归零），以达到减少过拟合的效果
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            # TODO mask是什么: 掩码张量，用于忽略特定位置的输入数据（例如填充位置），确保模型只关注有效的输入部分
            # 在处理变长序列时（例如不同长度的句子），为了将它们批量处理
            # x的形状：(batch_size, seq_len, embedding_dim)
            # mask的形状：(batch_size, seq_len)。每个元素的值为True或False，True代表该位置有效
            (MultiHeadAttention(self.token_emb_dim,
                                self.token_emb_dim,
                                self.token_emb_dim,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),
            # 归一化
            nn.LayerNorm(self.news_dim),

            nn.Dropout(p=cfg.dropout_probability),
            (AttentionPooling(self.news_dim,
                              cfg.model.attention_hidden_dim), 'x,mask -> x'),
            nn.LayerNorm(self.news_dim),
            # nn.Linear(self.news_dim, self.news_dim),
            # nn.LeakyReLU(0.2),
        ])


    def forward(self, key_entity_input, key_entity_input_mask):
        # batch_size, num_news = key_entity_input.shape[0], key_entity_input.shape[1]
        # print(f"key_entity_input: {key_entity_input}")
        # mask = (key_entity_input.sum(dim=2) == 0)
        # print(f"mask: {mask}")
        # print(f"key_entity_input.shape: {key_entity_input.shape}")
        # print(f"key_entity_input_mask.shape: {key_entity_input_mask.shape}")
        result = self.attention(key_entity_input, key_entity_input_mask)
        # print(f"result.shape: {result.shape}")
        return result