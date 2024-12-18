import torch
from torch import nn
from torch_geometric.nn import Sequential

from src.models.base.layers import MultiHeadAttention, AttentionPooling
from src.models.component.HieEntityEncoder import HieEntityEncoder


class HieEncoder(nn.Module):
    def __init__(self, cfg, news_encoder, entity_emb, category_encoder, subcategory_encoder):
        super().__init__()

        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = 400
        token_emb_dim = 400
        self.title_size = cfg.model.title_size
        self.entity_per_news = 5
        self.topic_per_user = cfg.model.topic_per_user
        self.news_per_topic = cfg.model.news_per_topic
        self.subtopic_per_user = cfg.model.subtopic_per_user
        self.news_per_subtopic = cfg.model.news_per_subtopic
        self.hie_news_encoder = news_encoder

        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        self.hie_entity_encoder = HieEntityEncoder(cfg, entity_emb)

        # self.atte = Sequential('x, mask', [
        #     (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
        #
        #     (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim,
        #                         int(self.entity_dim / cfg.model.head_dim), cfg.model.head_dim), 'x,x,x,mask -> x'),
        #     nn.LayerNorm(self.entity_dim),
        #     nn.Dropout(p=cfg.dropout_probability),
        #
        #     (AttentionPooling(self.entity_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
        #     nn.LayerNorm(self.entity_dim),
        #     nn.Linear(self.entity_dim, self.news_dim),
        #     nn.LeakyReLU(0.2),
        # ])

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

    def forward(self, clicked_topic_list, clicked_topic_mask_list,
                            clicked_subtopic_list, clicked_subtopic_mask_list, clicked_subtopic_news_list, clicked_subtopic_news_mask_list):
        # print(f"clicked_topic_list.shape: {clicked_topic_list.shape}") # [32, 8]
        # print(f"clicked_topic_mask_list.shape: {clicked_topic_mask_list.shape}") # [32, 8]
        # print(f"clicked_topic_news_list.shape: {clicked_topic_news_list.shape}") # [32, 8, 5, 43]
        # print(f"clicked_topic_news_mask_list.shape: {clicked_topic_news_mask_list.shape}") # [32, 8, 5]
        # print(f"clicked_subtopic_mask_list: {clicked_subtopic_mask_list}")
        # batch_size, topic_num, news_per_topic, _ = clicked_topic_news_list.shape
        # _, subtopic_num, news_per_subtopic, _ = clicked_subtopic_news_list.shape
        subtopic_news_input, subtopic_entity_input, _, _, _, _ = clicked_subtopic_news_list.split([self.title_size, 5, 1, 1, 1, 5], dim=-1)
        subtopic_news_mask = clicked_subtopic_news_mask_list.view(-1, self.news_per_subtopic)

        # print(f"subtopic_news_mask.shape: {subtopic_news_mask.shape}")
        # print(f"topic_news_input.shape: {topic_news_input.shape}") # [32, 8, 5, 30]
        # print(f"topic_entity_input.shape: {topic_entity_input.shape}") # [32, 8, 5, 5]

        # subtopic_news_input = subtopic_news_input.long().view(-1, self.news_per_subtopic, self.title_size)

        # print(f"subtopic_news_input.shape: {subtopic_news_input.shape}")  # [480, 5, 30]  480=32*15
        # subtopic_entity_input = subtopic_entity_input.long().view(-1, self.news_per_subtopic, self.entity_per_news)

        # print(f"subtopic_entity_input.shape: {subtopic_entity_input.shape}") # [480, 5, 5]


        subtopic_news_emb = self.hie_news_encoder(subtopic_news_input, subtopic_news_mask, None)
        # print(f"subtopic_news_emb.shape: {subtopic_news_emb.shape}") # [32, 400]
        # print(f"subtopic_entity_input.shape: {subtopic_entity_input.shape}") # [480, 5, 5]
        # print(f"subtopic_entity_input: {subtopic_entity_input}")
        subtopic_entity_emb = self.hie_entity_encoder(subtopic_entity_input)
        # print(f"subtopic_entity_emb.shape: {subtopic_entity_emb.shape}") # [32, 15, 400]
        subtopic_news_entity_emb = subtopic_news_emb + subtopic_entity_emb
        # print(f"subtopic_news_entity_emb.shape: {subtopic_news_entity_emb.shape}") # [32, 15, 400]
        # print(f"clicked_subtopic_list.shape: {clicked_subtopic_list.shape}") # [32, 15]
        subtopic_emb = self.subcategory_encoder(clicked_subtopic_list)
        # print(f"subtopic_emb.shape: {subtopic_emb.shape}") # [32, 15, 400]
        final_subtopic_emb = subtopic_news_entity_emb + subtopic_emb
        # print(f"final_subtopic_emb.shape: {final_subtopic_emb.shape}") # [32, 15, 400]
        final_subtopic_emb = self.attention(final_subtopic_emb, clicked_subtopic_mask_list)
        # print(f"final_subtopic_emb.shape: {final_subtopic_emb.shape}") # [32, 400]

        topic_emb = self.category_encoder(clicked_topic_list)
        # print(f"topic_emb.shape: {topic_emb.shape}") # [32, 8, 400]
        # print(f"topic_emb.shape: {topic_emb.shape}")
        # print(f"clicked_topic_mask_list.shape: {clicked_topic_mask_list.shape}")
        final_topic_emb = self.attention(topic_emb, clicked_topic_mask_list)
        # print(f"topic_emb.shape: {topic_emb.shape}") # [32, 400]
        total_emb = final_subtopic_emb + final_topic_emb
        # print(f"total_emb.shape: {total_emb.shape}") # [32, 400]
        return total_emb







