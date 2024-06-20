import numpy as np
import torch.nn as nn
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.nn import Sequential

from models import *
# from component import *
from src.models.base.layers import MultiHeadAttention, AttentionPooling
from src.models.component.subcategory_encoder import SubcategoryEncoder



class EventEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        # token_emb_dim = cfg.model.word_emb_dim
        token_emb_dim = 300
        self.event_dim = cfg.model.head_num * cfg.model.head_dim
        self.element_dim = 200
        self.news_dim = 400
        self.subcategory_embedding_dim = 200
        self.event_type_dim = 100

        if cfg.dataset.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            self.word_encoder = Sequential('x', [
                (nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0), 'x->x'),
            ])
        else:
            self.word_encoder = nn.Embedding(glove_emb+1, 300, padding_idx=0)
            nn.init.uniform_(self.word_encoder.weight, -1.0, 1.0)

        self.attention = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            # (MultiHeadAttention(token_emb_dim,
            #                     token_emb_dim,
            #                     token_emb_dim,
            #                     cfg.model.head_num,
            #                     cfg.model.head_dim), 'x,x,x,mask -> x'),
            (MultiHeadAttention(300,
                                300,
                                300,
                                20,
                                15), 'x,x,x,mask -> x'),
            nn.LayerNorm(300),

            nn.Dropout(p=cfg.dropout_probability),
            (AttentionPooling(300,
                              300), 'x,mask -> x'),
            nn.LayerNorm(300),
        ])

        self.event_type_encoder = Sequential('x', [
            (nn.Embedding(cfg.model.event_dict_size+2, self.event_type_dim), 'x->x'),
            (nn.Linear(self.event_type_dim, 300), 'x->x'),
            (nn.LeakyReLU(0.2), 'x->x')
        ])

        self.subcategory_encoder = Sequential('x', [
            (nn.Embedding(cfg.model.subcategory_size + 1, self.event_type_dim), 'x->x'),
            (nn.Linear(self.event_type_dim, 300), 'x->x'),
            (nn.LeakyReLU(0.2), 'x->x')
        ])

        self.event_transfer_encoder = EventTransferEncoder()


    def forward(self, event_input, event_mask):
        # print(f"event_input.shape = {event_input.shape}") # [32, 50, 11]

        batch_size = event_input.shape[0]
        num_news = event_input.shape[1]

        # nltk_event_features = event_type, event_entity, triggers, category, subcategory, news_index
        # event_input = np.concatenate([x for x in nltk_event_features], axis=1)
        # 1位事件类型索引、5位事件实体、3位triggers、1位category、1位subcategory、1位news_index
        # print(f"event_input.shape: {event_input.shape}")
        # TODO 解决event_entity、triggers中的零行问题
        event_type, event_entity, triggers, category, subcategory = event_input.split([1, 5, 3, 1, 1], dim=-1)
        # print(f"event_type.shape: {event_type.shape}")
        # print(f"event_entity.shape: {event_entity.shape}")
        # print(f"triggers.shape: {triggers.shape}")
        # print(f"category.shape: {category.shape}")
        # print(f"subcategory.shape: {subcategory.shape}")
        # print(f"event_entity: {event_entity}")
        # print(f"event_entity = {event_entity}")
        # event_entity_mask = event_entity != 0
        # print(f"event_entity_mask = {event_entity_mask}")
        # print(f"event_entity.shape: {event_entity.shape}")
        # print(f"triggers.shape: {triggers.shape}")
        # filtered_event_entity = event_entity[~(event_entity == 0).all(1)]
        # print(f"event_entity.shape = {event_entity.shape}")
        # print(f"filtered_event_entity.shape = {filtered_event_entity.shape}")
        # print(f"filtered_event_entity: {filtered_event_entity}")
        #
        # print(f"entity_mask.shape: {entity_mask.shape}")

        # TODO
        entity_mask = event_entity == 0 # [32, 50, 5]
        entity_emb = self.attention(self.word_encoder(event_entity.long().view(-1, 5)), entity_mask.view(-1, 5)).view(batch_size, num_news, 300)

        # print(f"entity attention finish.")
        # entity_emb = self.attention(entity_emb, None)
        # print(f"entity_emb = {entity_emb}")
        # entity_mask = ~(entity_word_emb == 0).all(dim=2)
        # filtered_event_entity = [sample[entity_mask[i]] for i, sample in enumerate(entity_word_emb)]
        # entity_word_emb = torch.nn.utils.rnn.pad_sequence(filtered_event_entity)

        # print(f"word_encoder finish.")
        # print(f"entity_word_emb.shape = {entity_word_emb.shape}")
        # print(f"entity_word_emb = {entity_word_emb}")
        # print(f"filtered_event_entity = {filtered_event_entity}")

        # entity_emb = self.attention(entity_word_emb, None)

        # print(f"entity_emb finish.")
        subcategory_emb = self.subcategory_encoder(subcategory).squeeze(-2)
        # print(f"subcategory attention finish.")
        # subcategory_emb = self.subcategory_encoder(subcategory)
        # print(f"subcategory_emb = {subcategory_emb}")
        # subcategory_emb = subcategory_emb[~(subcategory_emb == 0).all(1)]
        # subcategory_mask = ~(subcategory_emb == 0).all(dim=2)
        # subcategory_emb = subcategory_emb[subcategory_mask]
        # print(f"subcategory_emb = {subcategory_emb}")

        # category_emb = self.subcategory_encoder(100)
        # triggers_mask = triggers != 0
        # print(f"triggers = {triggers}")
        # print(f"triggers_mask = {triggers_mask}")
        # triggers = triggers[~(triggers == 0).all(1)]

        # TODO
        triggers_mask = triggers == 0
        triggers_emb = self.attention(self.word_encoder(triggers.long().view(-1, 3)), triggers_mask.view(-1, 3)).view(batch_size, num_news, 300)


        # print("trigger emb finish.")
        # triggers_emb = self.attention(triggers_emb, None)
        # print(f"triggers_emb = {triggers_emb}")
        # print(f"triggers_emb = {triggers_emb}")
        # triggers_total_emb = self.attention(triggers_emb, None)
        # print(f"triggers_total_emb = {triggers_total_emb}")

        # TODO
        event_type_emb = self.event_type_encoder(event_type).squeeze(-2)


        # print("event type emb finish.")
        # event_type_emb = self.event_type_encoder(event_type)
        # print(f"event_type_emb = {event_type_emb}")
        # event_type_mask = ~(event_type_emb == 0).all(dim=2)
        # event_type_emb = event_type_emb[event_type_mask]

        # print(f"event_type_emb shape: {event_type_emb.shape}")
        # print(f"entity_emb shape: {entity_emb.shape}")
        # print(f"triggers_emb shape: {triggers_emb.shape}")
        # print(f"subcategory_emb shape: {subcategory_emb.shape}")

        event_emb = torch.cat((event_type_emb, entity_emb, triggers_emb, subcategory_emb), dim=-1)
        # event_emb = subcategory_emb
        # print("event emb finish.")
        # print(f"event_emb shape: {event_emb.shape}")
        # print(f"event_emb: {event_emb}")
        result = self.event_transfer_encoder(event_emb)
        # print(f"result shape: {result.shape}") [1, 32, 400]
        return result




class EventTransferEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1200, 400)
        self.fc2 = nn.Linear(400, 400)

    def forward(self, event_input):
        x = torch.relu(self.fc1(event_input))
        result = self.fc2(x)
        return result

#
# list = [   17, 0,   0,   290,     0,     0, 15609,     0,     0,     3,     3]
# event_input = torch.from_numpy(np.array(list))
# event_encoder = EventEncoder(cfg)
# event_emb = event_encoder(event_input)
# print(event_emb)

class EventAttentionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.event_attention = Sequential('x, mask', [
            # dropout专门用于训练，推理阶段要关掉dropout
            # dropout：在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态（归零），以达到减少过拟合的效果
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(400,
                                400,
                                400,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),
            # 归一化
            nn.LayerNorm(400),

            nn.Dropout(p=cfg.dropout_probability),
            (AttentionPooling(400,
                              400), 'x,mask -> x'),
            nn.LayerNorm(400),
            # nn.Linear(400, 400),
            # nn.LeakyReLU(0.2),
        ])


    def forward(self, event_input, event_mask=None):
        batch_size, num_news, dim = event_input.shape
        # print(f"event attention encoder: event_input.shape: {event_input.shape}") # [32, 50, 400]
        result = self.event_attention(event_input, event_mask)
        # print(f"event attention encoder: result.shape: {result.shape}") # [32, 400]
        return result


class EventTotalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(800, 600)
        self.fc2 = nn.Linear(600, 400)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, event_encoder_input, event_attention_input):
        # batch_size, num_news= event_encoder_input.shape
        event_emb = torch.cat((event_encoder_input, event_attention_input), dim=-1)
        x = self.relu(self.fc1(event_emb))
        result = self.fc2(x)
        # print(f"event total encode result.shape: {result.shape}")
        return result
