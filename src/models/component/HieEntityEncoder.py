import torch
import torch.nn as nn
from models.base.layers import *
from torch_geometric.nn import Sequential

from src.models.base.layers import AttentionPooling, MultiHeadAttention


class HieEntityEncoder(nn.Module):
    def __init__(self, cfg, entity_emb):
        super().__init__()

        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = 400
        self.entity_emb_dim = 400
        pretrain = torch.from_numpy(entity_emb).float()
        self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

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

        self.atte2 = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_emb_dim, self.entity_emb_dim, self.entity_emb_dim,
                                int(self.entity_emb_dim / cfg.model.head_dim), cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.entity_emb_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(self.entity_emb_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(self.entity_emb_dim),
            nn.Linear(self.entity_emb_dim, self.entity_emb_dim),
            nn.LeakyReLU(0.2),
        ])



    def forward(self, entity_input, entity_mask=None):
        # entity_input: [batch_size, subtopic/topic num, entity_num, entity_index]
        # print(f"entity_input.shape: {entity_input.shape}")
        batch_size, num, entity_num, _ = entity_input.shape
        # print(f"entity input shape: {entity_input.shape}")
        # print(f"entity_input.dtype: {entity_input.dtype}")
        # entity_input = entity_input.float()
        entity_index = entity_input.long().view(-1, entity_num, entity_num)
        entity_emb = self.entity_embedding_layer(entity_index)
        # print(f"entity_emb shape: {entity_emb.shape}") # [batch_size * num, entity_num, 100]

        entity_emb = self.atte(entity_emb, entity_mask)
        # print(f"entity_emb shape: {entity_emb.shape}")  # [batch_size * num, 400]
        # print(f"entity_mask.shape: {entity_mask.shape}")
        # subtopic_entity_emb = self.atte2(entity_emb.view(batch_size, num, -1), entity_mask)
        # print(f"subtopic_entity_emb shape: {subtopic_entity_emb.shape}")
        # if entity_mask is not None:
            # print(f"entity_mask.dtype: {entity_mask.dtype}")
            # entity_mask = entity_mask.float()
            # result = self.atte(entity_input.view(batch_size*num, entity_num, self.entity_dim), entity_mask.view(batch_size*num, entity_num)).view(batch_size, num, self.entity_dim)
        # else:
            # print(f"entity_input.shape: {entity_input.shape}")
            # result = self.atte(entity_input.view(batch_size*num, entity_num, self.entity_dim), None).view(batch_size, num, self.entity_dim)
        # print(f"entity encoder result shape: {result.shape}")
        return entity_emb.view(batch_size, num, -1)