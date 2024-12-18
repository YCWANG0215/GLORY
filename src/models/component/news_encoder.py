import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path

import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *


class NewsEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        # word_emb_dim = 300, 为什么这么定义？
        # glove.840B.300 -> glove词向量维度是300
        token_emb_dim = cfg.model.word_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        if cfg.dataset.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            # padding_idx=0 -> 用于填充以确保所有输入序列有相同的长度
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:
            self.word_encoder = nn.Embedding(glove_emb+1, 300, padding_idx=0)
            nn.init.uniform_(self.word_encoder.weight, -1.0, 1.0)

        # 新闻标题、新闻摘要的最大长度
        self.view_size = [cfg.model.title_size, cfg.model.abstract_size]

        self.attention = Sequential('x, mask', [
            # dropout专门用于训练，推理阶段要关掉dropout
            # dropout：在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态（归零），以达到减少过拟合的效果
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            # TODO mask是什么: 掩码张量，用于忽略特定位置的输入数据（例如填充位置），确保模型只关注有效的输入部分
            # 在处理变长序列时（例如不同长度的句子），为了将它们批量处理
            # x的形状：(batch_size, seq_len, embedding_dim)
            # mask的形状：(batch_size, seq_len)。每个元素的值为True或False，True代表该位置有效
            (MultiHeadAttention(token_emb_dim,
                                token_emb_dim,
                                token_emb_dim,
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




    def forward(self, news_input, mask=None):
        """
        Args:
            news_input:  [batch_size, news_num, total_input]  eg. [64,50,82] [64,50,96]
            mask:   [batch_size, news_num]
        Returns:
            [batch_size, news_num, news_emb] eg. [64,50,400]
        """
        # print(f"news_input: {news_input.shape}") #[1, 15828, 43]
        batch_size = news_input.shape[0]
        num_news = news_input.shape[1]

        # TODO 为什么是news_input.split([self.view_size[0], 5, 1, 1, 1], dim=-1)
        # [batch_size * news_num, view_size, word_emb_dim]
        # print(f"news_input.shape: {news_input.shape}")
        title_input, _, _, _, _, _ = news_input.split([self.view_size[0], 5, 1, 1, 1, 5], dim=-1)
        # print(title_input.shape # [1, 15828, 30]
        title_word_emb = self.word_encoder(title_input.long().view(-1, self.view_size[0]))
        # print("news_encoder: ")
        # print(f"title_word_emb.shape: {title_word_emb.shape}") # [15828, 30, 300]
        # print(f"mask.shape: {mask.shape}")
        total_word_emb = title_word_emb

        result = self.attention(total_word_emb, mask)
        # print(f"news_encoder result: {result.shape}") # [15828, 400]
        return result.view(batch_size, num_news, self.news_dim)     # [batch, num_news, news_dim]
