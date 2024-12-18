from torch import nn
from models.base.layers import *
from torch_geometric.nn import Sequential



class SubcategoryEncoder(nn.Module):
    def __init__(self, subcategory_size):
        super().__init__()

        self.subcategory_emb_dim = 100
        self.news_emb_dim = 400
        self.embedding = nn.Embedding(subcategory_size+5, self.subcategory_emb_dim)
        self.fc = nn.Linear(self.subcategory_emb_dim, self.news_emb_dim)
        self.relu = nn.LeakyReLU(0.2)
        # print(f"subcategory_size: {subcategory_size}")

    def forward(self, subcategory_input):
        # print(f"subcategory_input: {subcategory_input}")
        # print(f"subcategory_input.shape: {subcategory_input.shape}")
        batch_size, num_news = subcategory_input.shape[0], subcategory_input.shape[1]
        # print(f"subcategory_encoder: subcategory_input.shape = {subcategory_input.shape}")
        # print(f"subcategory_encoder: batch_size={batch_size}, num_news={num_news}")
        embedding = self.embedding(subcategory_input)
        hidden = self.fc(embedding)
        # print(f"in subcategory_encoder, hidden.shape = {hidden.shape}")
        result = self.relu(hidden).view(batch_size, num_news, self.news_emb_dim)
        # print(f"in subcategory_encoder, result.shape = {result.shape}")

        return result

# class SubcategoryAttention(nn.Module):
#     def __init__(self, cfg, subcategory_size):
#         super().__init__()
#         self.subcategory_emb_dim = 400
#         self.news_emb_dim = 400
#         self.subcategory_encoder = SubcategoryEncoder(subcategory_size)
#
#         self.atte = Sequential('x', [
#             (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
#
#             (MultiHeadAttention(self.subcategory_emb_dim, self.subcategory_emb_dim, self.subcategory_emb_dim,
#                                 int(self.subcategory_emb_dim / cfg.model.head_dim), cfg.model.head_dim), 'x,x,x -> x'),
#             nn.LayerNorm(self.subcategory_emb_dim),
#             nn.Dropout(p=cfg.dropout_probability),
#
#             (AttentionPooling(self.subcategory_emb_dim, cfg.model.attention_hidden_dim), 'x -> x'),
#             nn.LayerNorm(self.subcategory_emb_dim),
#             nn.Linear(self.subcategory_emb_dim, self.news_emb_dim),
#             nn.LeakyReLU(0.2),
#         ])
#
#     def forward(self, subcategory_input):
#         batch_size, num_news, num_subcategories = subcategory_input.shape[0], subcategory_input.shape[1], subcategory_input.shape[2]
#         # print(f"subcategory_attention: batch_size, num_news, num_subcategories = {batch_size}, {num_news}, {num_subcategories}")
#         result = self.atte(self.subcategory_encoder(subcategory_input.view(batch_size * num_news, num_subcategories, 1))
#                            ).view(batch_size, num_news, self.news_emb_dim)
#         return result

# class GlobalSubcategoryEncoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.subcategory_emb_dim = 400
#         self.news_dim = cfg.model.head_num * cfg.model.head_dim
#
#         self.atte = Sequential('x', [
#             (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
#
#             (MultiHeadAttention(self.subcategory_emb_dim, self.subcategory_emb_dim, self.subcategory_emb_dim, cfg.model.head_num, cfg.model.head_dim), 'x,x,x -> x'),
#             nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
#             nn.Dropout(p=cfg.dropout_probability),
#
#             (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x -> x'),
#             nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
#         ])
#
#
#     def forward(self, subcategory_input):
#         batch_size, num_news = subcategory_input.shape[0], subcategory_input.shape[1]
#         result = self.atte(subcategory_input.view(batch_size * num_news, 1, self.news_dim)).view(batch_size, num_news, self.news_dim)
#         # print(f"result.shape = {result.shape}")
#         return result





