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


class HieNewsEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        # word_emb_dim = 300, 为什么这么定义？
        # glove.840B.300 -> glove词向量维度是300
        token_emb_dim = cfg.model.word_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.topic_per_user = cfg.model.topic_per_user
        self.news_per_topic = cfg.model.news_per_topic
        self.subtopic_per_user = cfg.model.subtopic_per_user
        self.news_per_subtopic = cfg.model.news_per_subtopic
        self.title_size = cfg.model.title_size

        if cfg.dataset.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            # padding_idx=0 -> 用于填充以确保所有输入序列有相同的长度
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:
            self.word_encoder = nn.Embedding(glove_emb+1, 300, padding_idx=0)
            nn.init.uniform_(self.word_encoder.weight, -1.0, 1.0)

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

        self.news_attention = Sequential('x, mask', [
            # dropout专门用于训练，推理阶段要关掉dropout
            # dropout：在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态（归零），以达到减少过拟合的效果
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            # TODO mask是什么: 掩码张量，用于忽略特定位置的输入数据（例如填充位置），确保模型只关注有效的输入部分
            # 在处理变长序列时（例如不同长度的句子），为了将它们批量处理
            # x的形状：(batch_size, seq_len, embedding_dim)
            # mask的形状：(batch_size, seq_len)。每个元素的值为True或False，True代表该位置有效
            (MultiHeadAttention(400,
                                400,
                                400,
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

    def forward(self, news_input, mask, subtopic_mask):
        """
        Args:
            news_input:  [batch_size, subtopic/topic num, news per subtopic/topic, title input]  eg. [32, 8, 5, 30]
            num: topic_num or subtopic_num
        Returns:
            [batch_size, topic/subtopic num, news_emb] eg. [32,8,400]
        """

        batch_size = news_input.shape[0]
        num = news_input.shape[1]
        news_num = news_input.shape[2]
        # title_input = news_input
        # print(f"news_input.shape: {news_input.shape}") # [32, 15, 5, 30]
        title_word = news_input.long().view(-1, news_num, self.title_size)
        total_word_emb = self.word_encoder(title_word)

        # print(f"total_word_emb.shape: {total_word_emb.shape}") # [480, 5, 30, 300]
        # print(f"mask.shape: {mask.shape}")
        total_word_emb = total_word_emb.view(-1, self.title_size, 300)
        # print(f"total_word_emb.shape: {total_word_emb.shape}") # [2400, 30, 300]
        title_emb = self.attention(total_word_emb, subtopic_mask)
        # print(f"title_emb.shape: {title_emb.shape}") # [2400, 400]
        title_emb = title_emb.view(-1, self.news_per_subtopic, self.news_dim)  # [32 * 15, 5, 400]
        news_emb = self.news_attention(title_emb, mask)
        # print(f"news_emb.shape: {news_emb.shape}") # [480, 400]
        news_emb = news_emb.view(-1, self.subtopic_per_user, self.news_dim)
        # print(f"news_emb.shape: {news_emb.shape}")  # [32, 15, 400]
        # print(f"subtopic_mask.shape: {subtopic_mask.shape}")
        # result = self.news_attention(news_emb, subtopic_mask)
        # print(f"result.shape: {result.shape}") # [32, 400]
        return news_emb.view(batch_size, num, -1)

