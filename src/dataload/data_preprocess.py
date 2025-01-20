import collections
import os
import subprocess
import tempfile
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch import scatter
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools

from utils.user_statistic_util import user_statistic


# from models.oneie.predict import *

def pad_to_fix_len(x, fix_length, padding_front=True, padding_value=0):
    if padding_front:
        pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
        mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
    else:
        pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
    return pad_x, np.array(mask, dtype='float32')

def update_dict(target_dict, key, value=None):
    """
    Function for updating dict with key / key+value

    Args:
        target_dict(dict): target dict
        key(string): target key
        value(Any, optional): if None, equals len(dict+1)
    """
    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)


def prepare_distributed_data(cfg, mode="train"):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # check
    target_file = os.path.join(data_dir[mode], f"behaviors_np{cfg.npratio}_0.tsv")
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target_file is not exist. New behavior file in {target_file}')

    behaviors = []
    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')

    if mode == 'train':
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t')
                impressions = [x.split('-') for x in imp.split(' ')]
                pos, neg = [], []
                for news_ID, label in impressions:
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:
                    neg_candidate = get_sample(neg, cfg.npratio)
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                    behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        # behaviors_per_file = [[] for _ in range(1)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)
            # behaviors_per_file[i % 1].append(line)

    elif mode in ['val', 'test']:
        # behaviors_per_file = [[] for _ in range(1)]
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                behaviors_per_file[i % cfg.gpu_num].append(line)
                # behaviors_per_file[i % 1].append(line)

    print(f'[{mode}]Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(data_dir[mode], f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)


# def run_command_and_process_json(command):
#     temp_fd, temp_path = tempfile.mkstemp(suffix='.json')
#     os.close(temp_fd)
#
#     try:
#         # 执行命令并输出重定向到临时文件
#         subprocess.run(command, shell=True, check=True, stdout=open(temp_path, 'w'), stderr=subprocess.PIPE)
#         # 读取临时JSON文件
#         with open(temp_path, 'r', encoding='utf-8') as temp_file:
#             data = json.load(temp_file)
#             # TODO 处理事件抽取模块得到的json文件
#             print(f"JSON data: {data}")
#     finally:
#         os.remove(temp_path)
#         print(f"Temp json file deleted.")

def read_raw_news(cfg, file_path, mode='train'):
    """
    Function for reading the raw news file, news.tsv

    Args:
        cfg:
        file_path(Path):                path of news.tsv
        mode(string, optional):        train or test


    Returns:
        tuple:     (news, news_index, category_dict, subcategory_dict, word_dict)


    """
    import nltk
    # nltk.download('punkt')
    
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}


    # TODO 数据预处理
    if mode in ['val', 'test']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        abs_entity_dict = pickle.load(open(Path(data_dir["train"]) / "abs_entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
        category_dict = pickle.load(open(Path(data_dir["train"]) / "category_dict.bin", "rb"))
        subcategory_dict = pickle.load(open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb"))
        events = pickle.load(open(Path(data_dir["train"]) / "events.bin", "rb"))
        event_dict = pickle.load(open(Path(data_dir["train"]) / "event_dict.bin", "rb"))
        event_type_dict = pickle.load(open(Path(data_dir["train"]) / "event_type_dict.bin", "rb"))
        # print(f"len(event_type_dict): {len(event_type_dict)}")
        # key_entity_dict = pickle.load(open(Path(data_dir["train"]) / "key_entity_dict.bin", "rb"))
        key_entities = pickle.load(open(Path(data_dir["train"]) / "key_entities.bin", "rb"))
        news_subtopic_map = pickle.load(open(Path(data_dir["train"]) / "news_subtopic_map.bin", "rb"))
        news_topic_map = pickle.load(open(Path(data_dir["train"]) / "news_topic_map.bin", "rb"))
        user_history_map = pickle.load(open(Path(data_dir["train"]) / "user_history_map.bin", "rb"))
        # key_entities_mask = pickle.load(open(Path(data_dir["train"]) / "key_entities_mask.bin", "rb"))
        # hetero_graph_map = pickle.load(open(Path(data_dir["train"]) / "hetero_graph_map.bin", "rb"))
        # role_dict = pickle.load(open(Path(data_dir["train"]) / "role_dict.bin", "rb"))
        node_dict = json.load(open(Path(data_dir["train"]) / "node_dict.json", "rb"))
        node_index = json.load(open(Path(data_dir["train"]) / "node_index.json", "rb"))

    else:
        news = {}
        news_dict = {}
        entity_dict = {}
        abs_entity_dict = {}
        category_dict = {}
        subcategory_dict = {}
        events = {}
        event_type_dict = {}
        event_dict = {}
        # key_entity_dict = {}
        key_entities = {}
        key_entities_mask = {}
        # role_dict = {}
        news_subtopic_map = {}
        news_topic_map = {}
        user_history_map = {}
        # hetero_graph_map = {}
        node_dict = {node_type: {} for node_type in ['topic', 'subtopic', 'argument', 'trigger', 'entity']}
        node_index = {node_type: {} for node_type in ['topic', 'subtopic', 'argument', 'trigger', 'entity']}


    # category_dict = {}
    # subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.
    # abs_word_cnt = Counter()

    num_line = len(open(file_path, encoding='utf-8').readlines())
    # behavior_num_line = len(open(file_path, encoding='utf-8').readlines())
    #
    # behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')
    # with open(behavior_file_path, 'r', encoding='utf-8') as f:
    #     for line in tqdm(f, total=behavior_num_line, desc=f"[{mode}]Processing user behavior"):
    #         line = line.strip().split('\t')
    #         click_id = line[3].split()[-self.cfg.model.his_size:]

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}]Processing raw news"):
            # split one line
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, abs_entity_str = split_line
            # print(f"t_entity_str = {t_entity_str}")
            update_dict(target_dict=news_dict, key=news_id)
            update_dict(target_dict=event_dict, key=news_id)


            # news_topic_map[news_id] = category
            # news_subtopic_map[news_id] = subcategory

            # provided_entity = []
            # provided_entity_abs = []
            # Entity

            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
                # print(f"{news_id}: {abs_entity_str}")
                abs_entity_ids = [obj["WikidataId"] for obj in json.loads(abs_entity_str)]
                # print(f"news_id: {news_id}: abs_entity_ids: {abs_entity_ids}")
                # print(f"abs_entity: {abs_entity_str}")
                # entity_ids += abs_entity_ids
                # [update_dict(target_dict=entity_dict, key=entity_id, value=node_dict['entity'][entity_id]) for entity_id in entity_ids]
                # for entity_id in entity_ids:
                #     print(f"{entity_id}: {node_dict['entity'][entity_id]}")
                [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
                [update_dict(target_dict=node_dict['entity'], key=entity_id, value=entity_dict[entity_id]) for entity_id in entity_ids]
                [update_dict(target_dict=node_index['entity'], key=entity_dict[entity_id], value=entity_id) for entity_id in entity_ids]
                # provided_entity = [obj["SurfaceForms"][0] for obj in json.loads(t_entity_str) if obj["SurfaceForms"]]
            else:
                entity_ids = t_entity_str


            # Abstract Entity
            # if abs_entity_str:
            #     abs_entity_ids = [obj["WikidataId"] for obj in json.loads(abs_entity_str)]
            #     [update_dict(target_dict=abs_entity_dict, key=abs_entity_id) for abs_entity_id in abs_entity_ids]
            #     provided_entity_abs = [obj["SurfaceForms"][0] for obj in json.loads(abs_entity_str) if obj["SurfaceForms"]]
            # else:
            #     abs_entity_ids = abs_entity_str



            # provided_entity += provided_entity_abs

            # TODO 事件抽取
            # news_entry: 标题+摘要
            # news_entry = f"{title}"
            # news_entry_dir = os.path.join(data_dir[f"{mode}"], "NewsTxt")
            event_json_dir = os.path.join(data_dir[f"{mode}"], "EventJson")
            # os.makedirs(news_entry_dir, exist_ok=True)
            # os.makedirs(event_json_dir, exist_ok=True)
            # news_entry_file_path = os.path.join(news_entry_dir, f"{news_id}.txt")

            # with open(news_entry_file_path, 'w') as news_entry_file:
            #     news_entry_file.write(news_entry + '\n')

            # TODO 读出事件抽取结果中的每个entities对应的token、triggers、relations以及roles
            event_json_path = os.path.join(event_json_dir, f"{news_id}.txt.json")
            # print(f"event_json_path: {event_json_path}")

            event_jsons = []
            with open(event_json_path, 'r', encoding='utf-8') as json_file:
                for json_line in json_file:
                    json_line = json_line.strip()
                    if not json_line:
                        continue
                    event_jsons.append(json.loads(json_line))

            event_entities = []
            for event_json in event_jsons:
                event_entities_origin = event_json["graph"]["entities"]
                for event_entity in event_entities_origin:
                    start, end, entity_type, entity_subtype, confidence = event_entity
                    entity_text = " ".join(event_json["tokens"][start:end]).lower()
                    update_dict(target_dict=node_dict['argument'], key=entity_text)
                    update_dict(target_dict=node_index['argument'], key=node_dict['argument'][entity_text], value=entity_text)
                    event_entities.append(entity_text)

                triggers_origin = event_json["graph"]["triggers"]
                triggers = []
                # # event_types = []
                # # event_type_confidence = []
                # for trigger in triggers_origin:
                #     start, end, _event_type, confidence = trigger
                #     trigger_text = " ".join(event_json["tokens"][start:end])
                #     triggers.append(trigger_text)
                #     # event_types.append(_event_type)
                #     # event_type_confidence.append(confidence)
                # event_type = f"{category}.{subcategory}"
                # event_types = []
                # event_type_confidence = []
                for trigger in triggers_origin:
                    start, end, _event_type, confidence = trigger
                    trigger_text = " ".join(event_json["tokens"][start:end]).lower()
                    update_dict(target_dict=node_dict['trigger'], key=trigger_text)
                    update_dict(target_dict=node_index['trigger'], key=node_dict['trigger'][trigger_text], value=trigger_text)
                    triggers.append(trigger_text)
                    # event_types.append(f"{category}.{subcategory}")
                    # event_types.append(_event_type)
                    # event_type_confidence.append(confidence)

                # NO
                # roles_origin = event_json["graph"]["roles"]
                # roles = []
                # for role in roles_origin:
                #     # role: 事件类型索引、实体索引、角色类型、置信度
                #     event_type_idx, entity_idx, role_type, confidence = role
                #     roles.append(role_type)
                #     update_dict(target_dict=role_dict, key=role_type)


            # if len(event_types) != 0:
            #     max_confidence = max(event_type_confidence)
            #     max_confidence_indices = [i for i, confidence in enumerate(event_type_confidence) if confidence == max_confidence]
            #     event_type = event_types[max_confidence_indices[0]]
            # else:
            #     event_type = f"Regular.{category}"
            #     event_type = f"{category}.{subcategory}"
            # print(f"event_type: {event_type}")

            # if len(event_types) != 0:
                # max_confidence = max(event_type_confidence)
                # max_confidence_indices = [i for i, confidence in enumerate(event_type_confidence) if
                #                           confidence == max_confidence]
                # event_type = event_types[max_confidence_indices[0]]
            # else:
            #     event_type = f"Regular.{category}"
            # print(f"event_type: {event_type}")
            event_type = f"{category}.{subcategory}"
            update_dict(target_dict=event_type_dict, key=event_type)
            # update_dict(target_dict=event_type_dict, key=event_types)




            # provided_entity_str = [' '.join(entity) for entity in provided_entity]
            # provided_entity_tuples = [tuple(entity) for entity in provided_entity]
            # event_entities_tuples = [tuple(entity) for entity in event_entities]
            # key_entity_tuples = list(set(provided_entity_tuples) & set(event_entities_tuples))


            # key_entity_list = []
            # key_entity_list = list(set(provided_entity) & set(event_entities))

            # key_entity = ' '.join(key_entity_list)
            # print(f"key_entity = {key_entity}")
            # key_entity_tokens = word_tokenize(key_entity.lower(), language=cfg.dataset.dataset_lang)

            # entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
            #                 [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
            #                 provided_entity = [obj["SurfaceForms"][0] for obj in json.loads(t_entity_str) if obj["SurfaceForms"]]
            # if key_entity_list:
            # print(f"type(key_entity) = {type(key_entity_list)}")


            # key_entity_ids = []
            # for key_entity in key_entity_list:
            #     for obj in json.loads(t_entity_str):
            #         # print(f"key_entity = {key_entity} , obj[SurfaceForms] = {sur}, id = {id}")
            #         if len(obj["SurfaceForms"]) > 0:
            #             sur = obj["SurfaceForms"][0]
            #         if key_entity == sur:
            #             id = obj["WikidataId"]
            #             key_entity_ids.append(id)

            # key_entity_ids = list(itertools.chain.from_iterable(key_entity_ids))
            # print(f"key_entity: {key_entity}")
            # print(f"key_entity_ids = {key_entity_ids}")



            # actual_key_entity_ids = []
            # print(f"key_entity_size: {cfg.model.key_entity_size}")
            # for key_entity_id in key_entity_ids:
            #     actual_key_entity_ids.append(entity_dict[key_entity_id])

            # print(f"key_entity_ids: {actual_key_entity_ids}, len: {len(actual_key_entity_ids)}")
            # if len(actual_key_entity_ids) >= cfg.model.key_entity_size:
            #     actual_key_entity_ids = actual_key_entity_ids[-cfg.model.key_entity_size:]
            # else:
            #     actual_key_entity_ids, actual_key_entity_mask = pad_to_fix_len(actual_key_entity_ids, cfg.model.key_entity_size)
            # print(len(actual_key_entity_ids))
            # print(f"after changed actual_key_entity_ids: {actual_key_entity_ids}")
            # update_dict(target_dict=key_entity_dict, key=news_id)
            # update_dict(target_dict=key_entities, key=news_id, value=[actual_key_entity_ids, actual_key_entity_mask])
            # news_idx = news_dict[news_id]
            # update_dict(target_dict=key_entities_mask, key=news_idx, value=actual_key_entity_mask)




            # key_entities = list(set(provided_entity) & set(event_entities))
            # key_entity = [list(entity) for entity in key_entity_tuples]
            # key_entity = [list(entity) for entity in key_entities]
            # print(f"provided_entity: {provided_entity}")
            # print(f"event_entities: {event_entities}")
            # print(f"key_entity: {key_entity}")
            # print()
            # update_dict(target_dict=key_entity_dict, key=news_id, value=key_entity_tokens)


            tokens = word_tokenize(title.lower(), language=cfg.dataset.dataset_lang)

            # abs_tokens = word_tokenize(abstract.lower(), language=cfg.dataset.dataset_lang)


            # event = {
            #     "event_type": event_type,
            #     "event_type_idx": event_type_dict[event_type],
            #     "event_entities": event_entities,
            #     # "entity_role": roles,
            #     "category": category,
            #     "subcategory": subcategory,
            #     "triggers": triggers
            # }

            # print(f"event: {event}")

            update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                                news_dict[news_id], abs_entity_ids])
            update_dict(target_dict=category_dict, key=category)
            update_dict(target_dict=node_dict['topic'], key=category, value=category_dict[category])
            # print(f"category: {category}, category_dict_value: {category_dict[category]}, node_dict: {node_dict['topic'][category]}")
            update_dict(target_dict=subcategory_dict, key=subcategory)
            update_dict(target_dict=node_dict['subtopic'], key=subcategory, value=subcategory_dict[subcategory])
            # print(f"subcategory: {subcategory}, subcategory_dict_value: {subcategory_dict[subcategory]}, node_dict: {node_dict['subtopic'][subcategory]}")

            update_dict(target_dict=events, key=news_id, value=[event_type, event_type_dict[event_type], event_entities, category, subcategory, triggers])
            # print(f"events: f{events[news_id]}")
            # update_dict(target_dict=events, key=news_id, value=event)
            update_dict(target_dict=news_subtopic_map, key=news_id, value=subcategory_dict[subcategory])
            update_dict(target_dict=news_topic_map, key=news_id, value=category_dict[category])
            # update_dict(target_dict=hetero_graph_map, key=news_id, value=[category, subcategory, triggers])
            if mode == 'train':
                word_cnt.update(tokens)
                # update_dict(target_dict=category_dict, key=category)
                # update_dict(target_dict=subcategory_dict, key=subcategory)
                # abs_word_cnt.update(abs_tokens)

        # TODO 事件抽取
        # with open(Path(data_dir[f"{mode}"]) / "news_idx.txt", 'w', encoding='utf-8') as idx_file:
        #     for idx in news_idx:
        #         idx_file.write(idx + '\n')
        #     print(f"[{mode}] news_idx.txt finish.")
        #
        # with open(Path(data_dir[f"{mode}"]) / "news_titles.txt", 'w',  encoding='utf-8') as title_file:
        #     for title in news_titles:
        #         title_file.write(title + '\n')
        #     print(f"[{mode}] news_title.txt finish.")
        # with open(Path(data_dir[f"{mode}"]) / "news_titles_abstracts.txt", 'w', encoding='utf-8') as title_abs_file:
        #     for entry in news_titles_abstracts:
        #         title_abs_file.write(entry + '\n')
        #     print(f"[{mode}] news_title_abstract.txt finish.")

        user_history_map = read_user_info(cfg, mode, news, news_dict, category_dict, subcategory_dict, user_history_map)

        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num]
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            # return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
            return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict, abs_entity_dict, event_type_dict, events, event_dict, key_entities, news_subtopic_map, news_topic_map, user_history_map, node_dict, node_index
        else:  # val, test
            # TODO 非训练集要不要返回category_dict、subcategory_dict
            # return news, news_dict, None, None, entity_dict, None

            return news, news_dict, category_dict, subcategory_dict, entity_dict, None, abs_entity_dict, event_type_dict, events, event_dict, key_entities, news_subtopic_map, news_topic_map, user_history_map, node_dict, node_index


def read_user_info(cfg, mode, news, news_dict, category_dict, subcategory_dict, user_history_map):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')
    behavior_num_line = len(open(behavior_file_path, encoding='utf-8').readlines())

    with open(behavior_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=behavior_num_line, desc=f"[{mode}]Processing user behavior"):
            line = line.strip().split('\t')
            user_id = line[1]
            if user_id in user_history_map:
                continue
            click_ids = line[3].split()[-cfg.model.his_size:]
            # print(f"click_ids: {click_ids}")
            topic_group = {}
            subtopic_group = {}

            for clicked_news in click_ids:
                news_info = news[clicked_news]
                # print(f"user {user_id}'s news_info: {news_info}")
                topic_id = category_dict[news_info[1]]
                subtopic_id = subcategory_dict[news_info[2]]

                if topic_id not in topic_group:
                    topic_group[topic_id] = []
                topic_group[topic_id].append(clicked_news)

                if subtopic_id not in subtopic_group:
                    subtopic_group[subtopic_id] = []
                subtopic_group[subtopic_id].append(clicked_news)

            update_dict(target_dict=user_history_map, key=user_id, value=[topic_group, subtopic_group])

    # TODO 用户信息统计
    # user_statistic(cfg, mode, user_history_map)

    return user_history_map



def read_parsed_news(cfg, news, news_dict,
                     category_dict=None, subcategory_dict=None, entity_dict=None,
                     word_dict=None, abs_entity_dict=None):
    # TODO 各个数据的维度： 标题（30）、实体（5）、类别（1）、子类别（1）、新闻索引（1）、摘要实体（5）
    news_num = len(news) + 1
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    news_entity = np.zeros((news_num, 5), dtype='int32')
    news_abs_entity = np.zeros((news_num, 5), dtype='int32')
    news_title = np.zeros((news_num, cfg.model.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        # _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]
        _title, _category, _subcategory, _entity_ids, _news_index, _abs_entity_ids = news[_news_id]

        # TODO 拿到category、subcategory、event_type
        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]



        # entity
        entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
        news_entity[_news_index, :min(cfg.model.entity_size, len(_entity_ids))] = entity_index[:cfg.model.entity_size]

        # abs_entity
        abs_entity_index = [abs_entity_dict[abs_entity_id] if abs_entity_id in abs_entity_dict else 0 for abs_entity_id in _abs_entity_ids]
        news_abs_entity[_news_index, :min(cfg.model.entity_size, len(_abs_entity_ids))] = abs_entity_index[:cfg.model.entity_size]

        # 处理标题 把标题中的每个单词映射到一个词典中的索引
        # TODO word_dict从哪传
        for _word_id in range(min(cfg.model.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    # TODO 传回新闻格式多了news_abs_entity
    return news_title, news_entity, news_category, news_subcategory, news_index, news_abs_entity


def read_news_events(cfg, news, news_dict, event_type_dict=None, events=None, word_dict=None, category_dict=None, subcategory_dict=None, event_dict=None, node_dict=None, node_index=None):
    # TODO 1位事件类型索引、5位事件实体、3位triggers、1位category、1位subcategory、1位news_index
    news_num = len(news) + 1
    category, subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    event_type = np.zeros((news_num, 1), dtype='int32')
    event_entity = np.zeros((news_num, 5), dtype='int32')
    triggers = np.zeros((news_num, 3), dtype='int32')
    # event_role = np.zeros((news_num, 3), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing news events"):
        # events: key=news_id, value=[event_type, event_type_dict[event_type], event_entities, category, subcategory, triggers])
        event_info = events[_news_id]
        # print(f"event_info = {event_info}")
        _, _category, _subcategory, _, _news_index, _abs_entity_ids = news[_news_id]
        event_type[_news_index, 0] = event_info[1]
        category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        # news_index[_news_index, 0] = news_dict[_news_id]

        # role_index = [role_dict[event_info[role_id]] if role_id in role_dict else 0 for role_id in event_info["entity_role"]]
        # event_role[_news_index, :min(3, len(event_info["entity_role"]))] = role_index[:3]
        for _entity_id in range(min(len(event_info[2]), 5)):
            if event_info[2][_entity_id] in word_dict:
                event_entity[_news_index, _entity_id] = word_dict[event_info[2][_entity_id]]

        for trigger_id in range(min(len(event_info[-1]), 3)):
            if event_info[-1][trigger_id] in word_dict:
                triggers[_news_index, trigger_id] = word_dict[event_info[-1][trigger_id]]



    # return event_type, event_entity, event_role, category, subcategory, news_index
    # return event_type, event_entity, triggers, category, subcategory, news_index
    # print(f"triggers: {triggers}")
    return event_type, event_entity, triggers, category, subcategory


def prepare_hetero_graph_info(cfg, node_dict, mode, news_dict):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    news_num = len(news_dict) + 1
    # print(f"news_num = {news_num}")
    topic, subtopic = [np.zeros((news_num, 1), dtype='int32') for _ in range(2)]
    triggers = np.zeros((news_num, 3), dtype='int32')
    arguments = np.zeros((news_num, 5), dtype='int32')
    entities = np.zeros((news_num, 5), dtype='int32')

    with open(Path(data_dir[mode]) / "hetero_graph_basic.json", 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            # hetero_graph_info = json.load(open(Path(data_dir[mode]) / "hetero_graph_basic.json"))
            # news_num = len(hetero_graph_info) + 1

            data = json.loads(line.strip())
            _news_id = data['news_id']
            _topic = data['topic']
            _subtopic = data['subtopic']
            _trigger = data['trigger']
            _argument = data['argument']
            _entity = data['entity']
            # print(f"news_id: {_news_id}")
            # print(f"topic: {_topic}")
            # print(f"_trigger: {_trigger}")
            # print(f"_argument: {_argument}")

            topic[news_dict[_news_id], 0] = node_dict["topic"][_topic]
            # print(f"topic: {_topic}, id: {topic[news_dict[_news_id], 0]}")
            subtopic[news_dict[_news_id], 0] = node_dict["subtopic"][_subtopic]
            # print(f"subtopic: {_subtopic}, id: {subtopic[news_dict[_news_id], 0]}")
            if _trigger == '[]':
                for i in range(3):
                    triggers[news_dict[_news_id], i] = 0
            else:
                _trigger = _trigger[1:-1].split(",")
                # print(f"_trigger: {_trigger}")
                trigger_ids = []
                for tri in _trigger:
                    tri = tri.strip()[1:-1].strip("'").lower()
                    # print(f"tri: {tri}")
                    if tri in node_dict["trigger"]:
                        trigger_ids.append(node_dict["trigger"][tri])
                # print(f"trigger_ids: {trigger_ids}")
                if len(trigger_ids) < 3:
                    for i in range(3 - len(trigger_ids)):
                        trigger_ids.append(0)
                # print(f"_trigger: {_trigger}")
                # print(f"trigger_ids: {trigger_ids}")
                # trigger_index = [node_dict["trigger"][trigger] if trigger in node_dict["trigger"] else 0 for trigger in _trigger]
                # print(f"trigger_index: {trigger_index}")
                # print(f"trigger_ids[:3]: {trigger_ids[:3]}")
                triggers[news_dict[_news_id], :min(3, len(trigger_ids))] = trigger_ids[:3]
                # print(f"news_dict[{_news_id}]: {news_dict[_news_id]}, triggers: {triggers[news_dict[_news_id], :min(3, len(trigger_ids))]}")
                # print(f"_triggers_: {triggers}")
            # print()
            # print(f"triggers: {triggers}")
            # print(f"trigger_index: {trigger_index}")
            if _argument == '[]':
                for i in range(5):
                    arguments[news_dict[_news_id], i] = 0
            else:
                _argument = _argument[1:-1].split(",")
                argument_ids = []
                for arg in _argument:
                    arg = arg.strip()[1:-1].strip("'").lower()
                    # print(f"arg: {arg}")
                    if arg in node_dict["argument"]:
                        argument_ids.append(node_dict["argument"][arg])
                # print(f"argument_ids: {argument_ids}")
                if len(argument_ids) < 5:
                    for i in range(5 - len(argument_ids)):
                        argument_ids.append(0)
                # print(f"argument_ids: {argument_ids}")
                # argument_index = [node_dict["argument"][argument] if argument in node_dict["argument"] else 0 for argument in _argument]
                # print(f"argument_index: {argument_index}")
                arguments[news_dict[_news_id], :min(5, len(argument_ids))] = argument_ids[:5]

            if _entity == '[]':
                for i in range(5):
                    entities[news_dict[_news_id], i] = 0
            else:
                _entity = _entity[1:-1].split(",")
                entity_ids = []
                for entity in _entity:
                    entity = entity.strip()[1:-1].strip("'")
                    if entity in node_dict["entity"]:
                        entity_ids.append(node_dict["entity"][entity])
                if len(entity_ids) < 5:
                    for i in range(5 - len(entity_ids)):
                        entity_ids.append(0)
                entities[news_dict[_news_id], :min(5, len(entity_ids))] = entity_ids[:5]
        # print(f"entities: {entities}")

        # for i in range(len(triggers)):
        #     print(f"news{i} triggers: {triggers[i, :3]}")

        # for i in range(len(arguments)):
        #     print(f"news{i} arguments: {arguments[i, :5]}")
        # print(f"triggers: {triggers}")
        # print(f"arguments: {arguments}")

    return topic, subtopic, triggers, arguments, entities






def prepare_preprocess_bin(cfg, mode):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if cfg.reprocess is True:
        # Glove
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict, abs_entity_dict, event_type_dict, events, event_dict, key_entities, news_subtopic_map, news_topic_map, user_history_map, node_dict, node_index = read_raw_news(
            file_path=Path(data_dir[mode]) / "news.tsv",
            cfg=cfg,
            mode=mode,
        )

        # TODO 是否限定train
        if mode == "train":
            # pickle.dump(category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb"))
            # pickle.dump(subcategory_dict, open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"))
            pickle.dump(word_dict, open(Path(data_dir[mode]) / "word_dict.bin", "wb"))
        else:
            # category_dict = pickle.load(open(Path(data_dir["train"]) / "category_dict.bin", "rb"))
            # subcategory_dict = pickle.load(open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb"))
            word_dict = pickle.load(open(Path(data_dir["train"]) / "word_dict.bin", "rb"))

        pickle.dump(entity_dict, open(Path(data_dir[mode]) / "entity_dict.bin", "wb"))
        pickle.dump(nltk_news, open(Path(data_dir[mode]) / "nltk_news.bin", "wb"))
        pickle.dump(nltk_news_dict, open(Path(data_dir[mode]) / "news_dict.bin", "wb"))
        pickle.dump(abs_entity_dict, open(Path(data_dir[mode]) / "abs_entity_dict.bin", "wb"))
        pickle.dump(category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb"))
        pickle.dump(subcategory_dict, open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"))
        pickle.dump(event_dict, open(Path(data_dir[mode]) / "event_dict.bin", "wb"))
        pickle.dump(event_type_dict, open(Path(data_dir[mode]) / "event_type_dict.bin", "wb"))
        pickle.dump(events, open(Path(data_dir[mode]) / "events.bin", "wb"))
        # pickle.dump(key_entity_dict, open(Path(data_dir[mode])/ "key_entity_dict.bin", "wb"))
        pickle.dump(key_entities, open(Path(data_dir[mode]) / "key_entities.bin", "wb"))
        pickle.dump(news_subtopic_map, open(Path(data_dir[mode]) / "news_subtopic_map.bin", "wb"))
        pickle.dump(news_topic_map, open(Path(data_dir[mode]) / "news_topic_map.bin", "wb"))
        pickle.dump(user_history_map, open(Path(data_dir[mode]) / "user_history_map.bin", "wb"))
        # pickle.dump(key_entities_mask, open(Path(data_dir[mode]) / "key_entities_mask.bin", "wb"))
        # pickle.dump(role_dict, open(Path(data_dir[mode]) / "role_dict.bin", "wb"))
        with open(Path(data_dir[mode]) / "node_dict.json", "w") as f:
            json.dump(node_dict, f, indent=4)
        with open(Path(data_dir[mode]) / "node_index.json", "w") as f:
            json.dump(node_index, f, indent=4)
        # nltk_news_features: news_title, news_entity, news_category, news_subcategory, news_index
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict, abs_entity_dict)
        # print(f"nltk_news_features[4] = {nltk_news_features[4]}")
        # def read_news_events(cfg, news, news_dict, event_dict=None, events=None, word_dict=None, role_dict=None, category_dict=None, subcategory_dict=None,):
        # TODO one line
        nltk_event_features = read_news_events(cfg, nltk_news, nltk_news_dict, event_type_dict, events, word_dict, category_dict, subcategory_dict, event_dict, node_dict, node_index)

        # TODO nltk_user_features
        # nltk_user_features = read_user_history()
        # news_input: 把输入的新闻特征信息连成一个大矩阵
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        # print(f"news_input.shape = {news_input.shape}") # [51283, 43]
        pickle.dump(news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb"))
        # TODO two line
        event_input = np.concatenate([x for x in nltk_event_features], axis=1)
        pickle.dump(event_input, open(Path(data_dir[mode]) / "nltk_token_event.bin", "wb"))

        hetero_graph_info = prepare_hetero_graph_info(cfg, node_dict, mode, nltk_news_dict)
        hetero_graph_news_input = np.concatenate([x for x in hetero_graph_info], axis=1)
        # print(f"hetero_graph_news_input.shape: {hetero_graph_news_input.shape}") # [51283, 15]
        pickle.dump(hetero_graph_news_input, open(Path(data_dir[mode]) / "hetero_graph_news_input.bin", "wb") )

        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')



def prepare_news_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return
    
    # -----------------------------------------News Graph------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    # news_dict: 新闻id+序号
    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    # nltk_token_news: 输入的新闻的所有信息链接成的大矩阵
    nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    
    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        # 读取behavior行数
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"):
                line = line.strip().split('\t')

                # check duplicate user
                used_id = line[1]
                if used_id in user_set:
                    continue
                else:
                    user_set.add(used_id)

                # record cnt & read path
                # 浏览过的新闻
                history = line[3].split()
                if len(history) > 1:
                    long_edge = [news_dict[news_id] for news_id in history]
                    edge_list.append(long_edge)

        # edge count
        node_feat = nltk_token_news
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):
            # Trajectory Graph 轨迹图
            if cfg.model.use_graph_type == 0:
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
                    # TODO 生成双向边（无向图？）
                    # short_edges.append((edge[i + 1], edge[i]))
            elif cfg.model.use_graph_type == 1:
                # Co-occurence Graph 共现图
                for i in range(len(edge) - 1):
                    for j in range(i+1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        edge_weights = Counter(short_edges)
        unique_edges = list(edge_weights.keys())

        # edge_index形状：(2, num_edges)，num_edges是边的数量。这个张量的第一行表示每条边的起点，第二行表示每条边的终点
        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        # 遍历每一条边，获取权重
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        # 创建图数据对象，包含节点特征、边索引和节点数量等信息
        data = Data(x=torch.from_numpy(node_feat),
                edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)

        # write_data_to_file(data, "news_graph.txt")
        # np.set_printoptions(threshold=np.inf)
        # print(data.x)

        torch.save(data, target_path)
        # print(data)
        print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
    
    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)
        
        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")


# def prepare_event_graph(cfg, mode='train'):
#     data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
#
#     nltk_target_path = Path(data_dir[mode]) / "nltk_event_graph.pt"
#
#     reprocess_flag = False
#     if nltk_target_path.exists() is False:
#         reprocess_flag = True
#
#     if (reprocess_flag == False) and (cfg.reprocess == False):
#         print(f"[{mode}] All graphs exist !")
#         return
#
#
#     # -----------------------------------------News Graph------------------------------------------------
#     behavior_path = Path(data_dir['train']) / "behaviors.tsv"
#     origin_graph_path = Path(data_dir['train']) / "nltk_event_graph.pt"
#
#     # news_dict: 新闻id+序号
#     news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
#     # event_dict: 事件类型id+序号
#     event_dict = pickle.load(open(Path(data_dir[mode]) / "event_dict.bin", "rb"))
#     # events: key->news_id, value: event
#     events = pickle.load(open(Path(data_dir[mode]) / "events.bin", "rb"))
#
#     # nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
#     nltk_token_event = pickle.load(open(Path(data_dir[mode])/ "nltk_token_event.bin", "rb"))
#
#     # ------------------- Build Graph -------------------------------
#     if mode == 'train':
#         edge_list, user_set = [], set()
#         # 读取behavior行数
#         num_line = len(open(behavior_path, encoding='utf-8').readlines())
#         with open(behavior_path, 'r', encoding='utf-8') as f:
#             for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to Event Graph"):
#                 line = line.strip().split('\t')
#
#                 # check duplicate user
#                 used_id = line[1]
#                 if used_id in user_set:
#                     continue
#                 else:
#                     user_set.add(used_id)
#
#                 # record cnt & read path
#                 # 浏览过的新闻
#                 history = line[3].split()
#                 if len(history) > 1:
#                     # long_edge = [news_dict[news_id] for news_id in history]
#                     # TODO long_edge存news_id?
#                     long_edge = [event_dict[news_id] for news_id in history]
#                     edge_list.append(long_edge)
#
#         # edge count
#         node_feat = nltk_token_event
#         target_path = nltk_target_path
#         num_nodes = len(news_dict) + 1
#
#         short_edges = []
#         for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing event edge list"):
#             # Trajectory Graph 轨迹图
#             if cfg.model.use_graph_type == 0:
#                 for i in range(len(edge) - 1):
#                     short_edges.append((edge[i], edge[i + 1]))
#                     # TODO 生成双向边（无向图？）
#                     # short_edges.append((edge[i + 1], edge[i]))
#             elif cfg.model.use_graph_type == 1:
#                 # Co-occurence Graph 共现图
#                 for i in range(len(edge) - 1):
#                     for j in range(i + 1, len(edge)):
#                         short_edges.append((edge[i], edge[j]))
#                         short_edges.append((edge[j], edge[i]))
#             else:
#                 assert False, "Wrong"
#
#         edge_weights = Counter(short_edges)
#         unique_edges = list(edge_weights.keys())
#
#         # edge_index形状：(2, num_edges)，num_edges是边的数量。这个张量的第一行表示每条边的起点，第二行表示每条边的终点
#         edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
#         # 遍历每一条边，获取权重
#         edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)
#
#         # 创建图数据对象，包含节点特征、边索引和节点数量等信息
#         data = Data(x=torch.from_numpy(node_feat),
#                     edge_index=edge_index, edge_attr=edge_attr,
#                     num_nodes=num_nodes)
#
#         # write_data_to_file(data, "news_graph.txt")
#         # np.set_printoptions(threshold=np.inf)
#         # print(data.x)
#
#         torch.save(data, target_path)
#         print(data)
#         print(f"[{mode}] Finish Event Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
#
#     elif mode in ['test', 'val']:
#         origin_graph = torch.load(origin_graph_path)
#         edge_index = origin_graph.edge_index
#         edge_attr = origin_graph.edge_attr
#         node_feat = nltk_token_event
#
#         data = Data(x=torch.from_numpy(node_feat),
#                     edge_index=edge_index, edge_attr=edge_attr,
#                     num_nodes=len(news_dict) + 1)
#
#         torch.save(data, nltk_target_path)
#         print(f"[{mode}] Finish nltk Event Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")




def write_data_to_file(data, file_path):
    with open(file_path, 'w') as f:
        f.write("Node Features:\n")
        f.write(str(data.x.numpy()) + "\n\n")
        f.write("Edge Index:\n")
        f.write(str(data.edge_index.numpy()) + "\n\n")
        f.write("Edge Attributes:\n")
        f.write(str(data.edge_attr.numpy()) + "\n\n")
        f.write("Number of Nodes: " + str(data.num_nodes) + "\n")


def prepare_neighbor_list(cfg, mode='train', target='news'):
    #--------------------------------Neighbors List-------------------------------------------
    print(f"[{mode}] Start to process neighbors list")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    neighbor_dict_path = Path(data_dir[mode]) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"{target}_weights_dict.bin"

    reprocess_flag = False
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] All {target} Neighbor dict exist !")
        return

    if target == 'news':
        print(f"preparing for new graph neighbor...")
        target_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'event':
        print(f"preparing for event graph neighbor...")
        target_graph_path = Path(data_dir[mode]) / "nltk_event_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "event_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'entity':
        print(f"preparing for entity graph neighbor...")
        target_graph_path = Path(data_dir[mode]) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'abs_entity':
        print(f"preparing for absolute entity graph neighbor...")
        target_graph_path = Path(data_dir[mode]) / "abs_entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "abs_entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'subcategory':
        print(f"preparing for subcategory graph neighbor...")
        target_graph_path = Path(data_dir[mode]) / "subcategory_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "subcategory_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    else:
        assert False, f"[{mode}] Wrong target {target} "

    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr

    if cfg.model.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    # TODO 构建每个结点的邻居字典和权重字典
    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)
    
    # for each node (except 0)
    for i in range(1, len(target_dict)+1):
        # 查找目标结点为i的所有边的边索引
        dst_edges = torch.where(edge_index[1] == i)[0]          # i as dst
        neighbor_weights = edge_attr[dst_edges]
        # edge_index[0]是边起始结点列表, neighbor_nodes拿到所有以结点i为目标节点的边的起始结点
        neighbor_nodes = edge_index[0][dst_edges]               # neighbors as src
        # 降序对边权重排序，返回排序后的权重和对应的索引
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
        # 根据排序后的索引更新邻居节点列表
        neighbor_dict[i] = neighbor_nodes[indices].tolist()
        neighbor_weights_dict[i] = sorted_weights.tolist()
    # TODO 上面是入度 要不要做出度？



    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print(f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")


def prepare_hetero_neighbor_list(cfg, mode='train'):
    print(f"[{mode}] Start to process hetero graph neighbors list")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    neighbor_dict_path = Path(data_dir[mode]) / f"hetero_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"hetero_weights_dict.bin"

    reprocess_flag = False
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Hetero Graph Neighbor dict exist !")
        return

    node_types = ['topic', 'subtopic', 'trigger', 'argument', 'entity']

    # meta_path_type = [
    #     ['entity', 'entity', 'entity'],
    #     ['entity', 'topic', 'entity'],
    #     ['entity', 'subtopic', 'entity'],
    #     ['entity', 'trigger', 'entity'],
    #     ['entity', 'argument', 'entity'],
    #     ['argument', 'entity'],
    #     ['entity', 'entity'],
    #     ['trigger', 'entity'],
    #     ['subtopic', 'entity'],
    #     ['topic', 'entity']
    # ]

    hetero_graph_path = Path(data_dir[mode]) / "hetero_graph.pt"
    origin_graph = torch.load(hetero_graph_path)
    node_index = json.load(open(Path(data_dir[mode]) / "node_index.json"))

    data = HeteroData()
    data['topic'].x = origin_graph['topic']
    data['subtopic'].x = origin_graph['subtopic']
    data['trigger'].x = origin_graph['trigger']
    data['argument'].x = origin_graph['argument']
    data['entity'].x = origin_graph['entity']

    data['entity', 'to', 'topic'].edge_index = origin_graph['entity', 'to', 'topic'].edge_index
    data['entity', 'to', 'subtopic'].edge_index = origin_graph['entity', 'to', 'subtopic'].edge_index
    data['entity', 'to', 'trigger'].edge_index = origin_graph['entity', 'to', 'trigger'].edge_index
    data['entity', 'to', 'argument'].edge_index = origin_graph['entity', 'to', 'argument'].edge_index
    data['entity', 'to', 'entity'].edge_index = origin_graph['entity', 'to', 'entity'].edge_index
    data['topic', 'to', 'entity'].edge_index = origin_graph['topic', 'to', 'entity'].edge_index
    data['subtopic', 'to', 'entity'].edge_index = origin_graph['subtopic', 'to', 'entity'].edge_index
    data['trigger', 'to', 'entity'].edge_index = origin_graph['trigger', 'to', 'entity'].edge_index
    data['argument', 'to', 'entity'].edge_index = origin_graph['argument', 'to', 'entity'].edge_index

    data['entity', 'to', 'topic'].edge_attr = origin_graph['entity', 'to', 'topic'].edge_attr
    data['entity', 'to', 'subtopic'].edge_attr = origin_graph['entity', 'to', 'subtopic'].edge_attr
    data['entity', 'to', 'trigger'].edge_attr = origin_graph['entity', 'to', 'trigger'].edge_attr
    data['entity', 'to', 'argument'].edge_attr = origin_graph['entity', 'to', 'argument'].edge_attr
    data['entity', 'to', 'entity'].edge_attr = origin_graph['entity', 'to', 'entity'].edge_attr
    data['topic', 'to', 'entity'].edge_attr = origin_graph['topic', 'to', 'entity'].edge_attr
    data['subtopic', 'to', 'entity'].edge_attr = origin_graph['subtopic', 'to', 'entity'].edge_attr
    data['trigger', 'to', 'entity'].edge_attr = origin_graph['trigger', 'to', 'entity'].edge_attr
    data['argument', 'to', 'entity'].edge_attr = origin_graph['argument', 'to', 'entity'].edge_attr

    edge_types = ['entity_topic', 'entity_subtopic', 'entity_trigger', 'entity_argument','entity_entity',
                  'topic_entity', 'subtopic_entity', 'trigger_entity', 'argument_entity']

    neighbor_dict = {edge_type: collections.defaultdict(list) for edge_type in edge_types}
    neighbor_weights_dict = {edge_type: collections.defaultdict(list) for edge_type in edge_types}


    # for src_node_type in node_types:
    #     # 源节点是entity，遍历所有类型的邻居
    #     for dst_node_type in node_types:
    #         if src_node_type != 'entity' and dst_node_type != 'entity':
    #             continue
    #         edge_type = f'{src_node_type}_{dst_node_type}'
    #         for i in range(1, len(node_index[dst_node_type]) + 1):
    #             # 查找目标结点为i的所有边索引 找起始结点为i的所有边索引？
    #             dst_edges = torch.where(data[src_node_type, 'to', dst_node_type].edge_index[1] == i)[0]
    #             neighbor_weights = data[src_node_type, 'to', dst_node_type].edge_attr[dst_edges]
    #             # 起始结点
    #             neighbor_nodes = data[src_node_type, 'to', dst_node_type].edge_index[0][dst_edges]
    #             sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
    #             neighbor_dict[edge_type][i] = neighbor_nodes[indices].tolist()
    #             neighbor_weights_dict[edge_type][i] = sorted_weights.tolist()

    for src_node_type in node_types:
        # 源节点是entity，遍历所有类型的邻居
        for dst_node_type in node_types:
            if src_node_type != 'entity' and dst_node_type != 'entity':
                continue
            edge_type = f'{src_node_type}_{dst_node_type}'
            for i in range(1, len(node_index[src_node_type]) + 1):
                # 找起始结点为i的所有边索引
                edges_idx = torch.where(data[src_node_type, 'to', dst_node_type].edge_index[0] == i)[0]
                neighbor_weights = data[src_node_type, 'to', dst_node_type].edge_attr[edges_idx]
                # 边的终点
                neighbor_nodes = data[src_node_type, 'to', dst_node_type].edge_index[1][edges_idx]
                sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
                neighbor_dict[edge_type][i] = neighbor_nodes[indices].tolist()
                neighbor_weights_dict[edge_type][i] = sorted_weights.tolist()

    pickle.dump(neighbor_dict, open(neighbor_dict_path, 'wb'))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, 'wb'))
    print(f"[{mode}] Finish Hetero Graph Neighbor dict.")


def update_entity_pool_dict(entity_pool_dict, entity_ids, entity_weights):
    for entity, weight in zip(entity_ids, entity_weights):
        if entity in entity_pool_dict:
            entity_pool_dict[entity] += weight
        else:
            entity_pool_dict[entity] = weight

    return entity_pool_dict

def build_direct_entity_pool(cfg, public_adjacent_pool, public_adjacent_weights, strong_adjacent_pool, strong_adjacent_weights):
    direct_entity_pool_dict = {}
    # update_entity_pool_dict(direct_entity_pool_dict, public_adjacent_pool, public_adjacent_weights)
    update_entity_pool_dict(direct_entity_pool_dict, strong_adjacent_pool, strong_adjacent_weights)

    sorted_pool = sorted(direct_entity_pool_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_key = [k for k, _ in sorted_pool]
    return sorted_key[:cfg.model.direct_entity_num]


def build_indirect_entity_pool(cfg, public_adjacent_pool, public_adjacent_weights, indirect_entity_pool, indirect_entity_weights, indirect_non_entity_pool, indirect_non_entity_weights):
    indirect_entity_pool_dict = {}
    update_entity_pool_dict(indirect_entity_pool_dict, public_adjacent_pool, public_adjacent_weights)
    update_entity_pool_dict(indirect_entity_pool_dict, indirect_entity_pool, indirect_entity_weights)
    update_entity_pool_dict(indirect_entity_pool_dict, indirect_non_entity_pool, indirect_non_entity_weights)

    sorted_pool = sorted(indirect_entity_pool_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_key = [k for k, _ in sorted_pool]
    return sorted_key[:cfg.model.indirect_entity_num]

def build_direct_and_indirect_entity_pool(cfg, mode='train'):
    print(f"[{mode}] Start to process direct and indirect entity pool")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    direct_entity_pool_path = Path(data_dir[mode]) / f"direct_entity_pool.bin"
    indirect_entity_pool_path = Path(data_dir[mode]) / f"indirect_entity_pool.bin"
    # direct_entity_pool_path = Path(data_dir[mode]) / f"direct_entity_pool.json"
    # indirect_entity_pool_path = Path(data_dir[mode]) / f"indirect_entity_pool.json"

    # hetero_graph_adjacent_pool_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_pool.json"
    # hetero_graph_adjacent_weights_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_weights.json"

    hetero_graph_adjacent_pool_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_pool.bin"
    hetero_graph_adjacent_weights_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_weights.bin"
    reprocess_flag = False
    for file_path in [direct_entity_pool_path, indirect_entity_pool_path]:
        if file_path.exists() is False:
            reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        # print(f"[{mode}] Hetero Graph Neighbor dict exist !")
        return

    hetero_graph_adjacent_pool = json.load(open(hetero_graph_adjacent_pool_path, 'rb'))
    hetero_graph_adjacent_weights = json.load(open(hetero_graph_adjacent_weights_path, 'rb'))

    # direct_entity_pool = {}
    # indirect_entity_pool = {}
    direct_entity_pool = [0]
    indirect_entity_pool = [0]

    # print(f"hetero_graph_adjacent_pool: {hetero_graph_adjacent_pool}")
    for cur_news_index in range(1, len(hetero_graph_adjacent_pool)):
        cur_news_index = f"{cur_news_index}"
        strong_adjacent_pool = hetero_graph_adjacent_pool[cur_news_index]['strong_adjacent_pool']
        strong_adjacent_weights = hetero_graph_adjacent_weights[cur_news_index]['strong_adjacent_weights']
        public_adjacent_pool = hetero_graph_adjacent_pool[cur_news_index]['public_adjacent_pool']
        public_adjacent_weights = hetero_graph_adjacent_weights[cur_news_index]['public_adjacent_weights']
        indirect_entity_adjacent_pool = hetero_graph_adjacent_pool[cur_news_index]['indirect_entity_adjacent_pool']
        indirect_entity_adjacent_weights = hetero_graph_adjacent_weights[cur_news_index][
            'indirect_entity_adjacent_weights']
        indirect_non_entity_adjacent_pool = hetero_graph_adjacent_pool[cur_news_index][
            'indirect_non_entity_adjacent_pool']
        indirect_non_entity_adjacent_weight = hetero_graph_adjacent_weights[cur_news_index][
            'indirect_non_entity_adjacent_weights']

        cur_direct_entity_pool = build_direct_entity_pool(cfg, public_adjacent_pool, public_adjacent_weights,
                                                           strong_adjacent_pool, strong_adjacent_weights)
        cur_indirect_entity_pool = build_indirect_entity_pool(cfg, public_adjacent_pool, public_adjacent_weights,
                                                               indirect_entity_adjacent_pool,
                                                               indirect_entity_adjacent_weights,
                                                               indirect_non_entity_adjacent_pool,
                                                               indirect_non_entity_adjacent_weight)
        # direct_entity_pool[cur_news_index] = cur_direct_entity_pool
        # indirect_entity_pool[cur_news_index)] = cur_indirect_entity_pool
        direct_entity_pool.append(cur_direct_entity_pool)
        indirect_entity_pool.append(cur_indirect_entity_pool)
    # with open(direct_entity_pool_path, 'w', encoding='utf-8') as f:
    #     json.dump(direct_entity_pool, f, ensure_ascii=False, indent=4)
    # with open(indirect_entity_pool_path, 'w', encoding='utf-8') as f:
    #     json.dump(indirect_entity_pool, f, ensure_ascii=False, indent=4)
    pickle.dump(direct_entity_pool, open(direct_entity_pool_path, 'wb'))
    pickle.dump(indirect_entity_pool, open(indirect_entity_pool_path, 'wb'))





def prepare_entity_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    if mode == 'train':

        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        # print("news_graph,", news_graph)

        # entity_indices：每个新闻中包含的实体的索引
        # TODO 多了5维abs_entity
        entity_indices = news_graph.x[:, -13:-8].numpy()
        # entity_indices = news_graph.x[:, -8:-3].numpy()
        # print("entity_indices, ", entity_indices.shape)

        entity_edge_index = []
        # -------- Inter-news -----------------
        # 同一个新闻内存在的实体连成边
        # for entity_idx in entity_indices:
        #     entity_idx = entity_idx[entity_idx > 0]
        #     edges = list(itertools.combinations(entity_idx, r=2))
        #     entity_edge_index.extend(edges)

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            # 过滤无效实体索引
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            # 生成实体对组合并加权
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)

        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
            
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
        
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")

def prepare_abs_entity_graph(cfg, mode='train'):
    print("Building Multi-User Entity Graph...")
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    target_path = Path(data_dir[mode]) / "abs_entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Multi-User Entity graph exists!")
        return
    abs_entity_dict = pickle.load(open(Path(data_dir[mode]) / "abs_entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "abs_entity_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        print("news_graph,", news_graph)
        # 多了5维
        # entity_indices = news_graph.x[:, -8:-3].numpy()
        abs_entity_indices = news_graph.x[:, -5:].numpy()
        print("abs_entity_indices, ", abs_entity_indices.shape)

        abs_entity_edge_index = []
        # news_edge_src, news_edge_dest = news_graph.edge_index
        # edge_weights = news_graph.edge_attr.long().tolist()
        # TODO 同一条新闻中的实体建双向图，表示实体间的强相关性
        for abs_entity_idx in abs_entity_indices:
            abs_entity_idx = abs_entity_idx[abs_entity_idx > 0]
            edges = list(itertools.combinations(abs_entity_idx, r=2))
            abs_entity_edge_index.extend(edges)
        
        edge_weights = Counter(abs_entity_edge_index)
        unique_edges = list(edge_weights.keys())
        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)
        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(abs_entity_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(abs_entity_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Abstract Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(abs_entity_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(abs_entity_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Abstract Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")

        # TODO 相邻新闻中的实体建单向图
        # for i in range(news_edge_src.shape[0]):
        #     src_entities = entity_indices[news_edge_src[i]]
        #     dest_entities = entity_indices[news_edge_dest[i]]
        #     # 过滤无效实体索引
        #     src_entities_mask = src_entities > 0
        #     dest_entities_mask = dest_entities > 0
        #     src_entities = src_entities[src_entities_mask]
        #     dest_entities = dest_entities[dest_entities_mask]
        #     # 生成实体对组合并加权
        #     edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
        #     multi_user_entity_edge_index.extend(edges)


def prepare_hetero_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    target_path = Path(data_dir[mode]) / "hetero_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] Hetero graph exists!")
        return
    news_set = set()
    # hetero_graph = HeteroData()

    node_types = ['trigger', 'argument', 'topic', 'subtopic', 'entity']
    edge_weights = collections.defaultdict(int)
    hetero_graph_basic_dict = {}
    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    node_dict = json.load(open(Path(data_dir[mode]) / "node_dict.json"))
    with open(Path(data_dir[mode]) / "hetero_graph_basic.json", "rb") as f:
        for line in f:
            data = json.loads(line.strip())
            news_id = data['news_id']
            if news_id in news_set:
                continue
            else:
                news_set.add(news_id)
                hetero_graph_basic_dict[news_id] = {}
            topic = data['topic']
            subtopic = data['subtopic']
            _argument = data['argument'][1:-1].split(",")
            _trigger = data['trigger'][1:-1].split(",")
            _entity = data['entity'][1:-1].split(",")

            topic_id = node_dict['topic'][topic]
            subtopic_id = node_dict['subtopic'][subtopic]
            argument, trigger, entity = [], [], []
            argument_ids, trigger_ids, entity_ids = [], [], []

            for __trigger in _trigger:
                __trigger = __trigger.strip()[1:-1].strip("").lower()
                if __trigger != "" and __trigger in node_dict['trigger']:
                    trigger.append(__trigger)
                    trigger_ids.append(node_dict['trigger'][__trigger])

            for __argument in _argument:
                __argument = __argument.strip()[1:-1].strip("").lower()
                if __argument != "" and __argument in node_dict['argument']:
                    argument.append(__argument)
                    argument_ids.append(node_dict['argument'][__argument])

            for __entity in _entity:
                __entity = __entity.strip()[1:-1].strip("")
                if __entity != "" and __entity in node_dict['entity']:
                    entity.append(__entity)
                    entity_ids.append(node_dict['entity'][__entity])

            # print(f"{news_id}: topic: {topic}({topic_id}), subtopic: {subtopic}({subtopic_id})")
            # print(f"{news_id}: argument: {argument}, trigger: {trigger}, entity: {entity}")
            # print(f"{news_id}: argument_id: {argument_ids}, trigger_ids: {trigger_ids}, entity_ids: {entity_ids}")
            hetero_graph_basic_dict[news_id]["topic"] = topic_id
            hetero_graph_basic_dict[news_id]["subtopic"] = subtopic_id
            hetero_graph_basic_dict[news_id]["argument"] = argument_ids
            hetero_graph_basic_dict[news_id]["trigger"] = trigger_ids
            hetero_graph_basic_dict[news_id]["entity"] = entity_ids
            # print(f"{news_id}: {hetero_graph_basic_dict[news_id]}")
    with open(Path(data_dir[mode]) / "hetero_basic_info", "w", encoding="utf-8") as f:
        json.dump(hetero_graph_basic_dict, f, ensure_ascii=False, indent=4)


    origin_graph_path = Path(data_dir['train']) / "hetero_graph.pt"
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"

    # ----------------------- Build Hetero Graph ---------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors to Hetero Graph"):
                line = line.strip().split('\t')

                user_id = line[1]
                if user_id in user_set:
                    continue
                else:
                    user_set.add(user_id)

                history = line[3].split()
                if len(history) > 1:
                    long_edge = [news_id for news_id in history]
                    edge_list.append(long_edge)

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f'Processing hetero graph edge list'):
            for i in range(len(edge)-1):
                short_edges.append((edge[i], edge[i+1]))
        news_edge_weights = Counter(short_edges)
        # print(f"edge_weights: {edge_weights}")
        unique_edges = list(news_edge_weights.keys())
        # print(f"unique_edges: {unique_edges}")
        news_edge_index = list(zip(*unique_edges))
        # print(f"news_edge_index: {news_edge_index}")
        news_edge_attr = [news_edge_weights[edge] for edge in unique_edges]
        # print(f"news_edge_attr: {news_edge_attr}")
        # print(f"len(news_edge_index): {len(news_edge_index[0])}") # 632044
        # print(f"len(news_edge_attr): {len(news_edge_attr)}")

        entity_topic_edges = []
        entity_subtopic_edges = []
        entity_trigger_edges = []
        entity_argument_edges = []
        entity_entity_edges = []
        topic_entity_edges = []
        subtopic_entity_edges = []
        trigger_entity_edges = []
        argument_entity_edges = []

        for i in range(len(news_edge_attr)):
            src_news = news_edge_index[0][i]
            dst_news = news_edge_index[1][i]
            weight = news_edge_attr[i]
            src_news_info = hetero_graph_basic_dict[src_news]
            dst_news_info = hetero_graph_basic_dict[dst_news]
            src_topic, src_subtopic, src_arguments, src_triggers, src_entities = (src_news_info[key] for key in ["topic", "subtopic", "argument", "trigger", "entity"])
            dst_topic, dst_subtopic, dst_arguments, dst_triggers, dst_entities = (dst_news_info[key] for key in ["topic", "subtopic", "argument", "trigger", "entity"])

            for src_entity in src_entities:
                for times in range(weight):
                    # src_entities -> dst_topic
                    entity_topic_edges.append((src_entity, dst_topic))
                    # topic_entity_edges.append((dst_topic, src_entity))
                    # src_entities -> dst_subtopic
                    entity_subtopic_edges.append((src_entity, dst_subtopic))
                    # subtopic_entity_edges.append((dst_subtopic, src_entity))
                    # src_entities -> dst_triggers
                    for dst_trigger in dst_triggers:
                        entity_trigger_edges.append((src_entity, dst_trigger))
                    # src_entities -> dst_arguments
                    for dst_argument in dst_arguments:
                        entity_argument_edges.append((src_entity, dst_argument))
                    # src_entities <-> dst_entities
                    for dst_entity in dst_entities:
                        entity_entity_edges.append((src_entity, dst_entity))

            for dst_entity in dst_entities:
                for times in range(weight):
                    # src_topic -> dst_entities
                    topic_entity_edges.append((src_topic, dst_entity))
                    # entity_topic_edges.append((dst_entity, src_topic))
                    # src_subtopic -> dst_entities
                    subtopic_entity_edges.append((src_subtopic, dst_entity))
                    # entity_subtopic_edges.append((dst_entity, src_subtopic))
                    # src_triggers -> dst_entities
                    for src_trigger in src_triggers:
                        trigger_entity_edges.append((src_trigger, dst_entity))
                    # src_arguments -> dst_entities
                    for src_argument in src_arguments:
                        argument_entity_edges.append((src_argument, dst_entity))
        # print(f"entity_topic_edges: {entity_topic_edges}")
        entity_topic_weights = Counter(entity_topic_edges)
        entity_subtopic_weights = Counter(entity_subtopic_edges)
        entity_trigger_weights = Counter(entity_trigger_edges)
        entity_argument_weights = Counter(entity_argument_edges)
        entity_entity_weights = Counter(entity_entity_edges)
        topic_entity_weights = Counter(topic_entity_edges)
        subtopic_entity_weights = Counter(subtopic_entity_edges)
        trigger_entity_weights = Counter(trigger_entity_edges)
        argument_entity_weights = Counter(argument_entity_edges)

        unique_entity_topic_edges = list(entity_topic_weights.keys())
        # print(f"unique_entity_topic_edges: {unique_entity_topic_edges}")
        unique_entity_subtopic_edges = list(entity_subtopic_weights.keys())
        unique_entity_trigger_edges = list(entity_trigger_weights.keys())
        unique_entity_argument_edges = list(entity_argument_weights.keys())
        unique_entity_entity_edges = list(entity_entity_weights.keys())
        unique_topic_entity_edges = list(topic_entity_weights.keys())
        unique_subtopic_entity_edges = list(subtopic_entity_weights.keys())
        unique_trigger_entity_edges = list(trigger_entity_weights.keys())
        unique_argument_entity_edges = list(argument_entity_weights.keys())

        entity_topic_edge_index = list(zip(*unique_entity_topic_edges))
        entity_subtopic_edge_index = list(zip(*unique_entity_subtopic_edges))
        entity_trigger_edge_index = list(zip(*unique_entity_trigger_edges))
        entity_argument_edge_index = list(zip(*unique_entity_argument_edges))
        entity_entity_edge_index = list(zip(*unique_entity_entity_edges))
        topic_entity_edge_index = list(zip(*unique_topic_entity_edges))
        subtopic_entity_edge_index = list(zip(*unique_subtopic_entity_edges))
        trigger_entity_edge_index = list(zip(*unique_trigger_entity_edges))
        argument_entity_edge_index = list(zip(*unique_argument_entity_edges))

        entity_topic_edge_attr = [entity_topic_weights[edge] for edge in unique_entity_topic_edges]
        entity_subtopic_edge_attr = [entity_subtopic_weights[edge] for edge in unique_entity_subtopic_edges]
        entity_trigger_edge_attr = [entity_trigger_weights[edge] for edge in unique_entity_trigger_edges]
        entity_argument_edge_attr = [entity_argument_weights[edge] for edge in unique_entity_argument_edges]
        entity_entity_edge_attr = [entity_entity_weights[edge] for edge in unique_entity_entity_edges]
        topic_entity_edge_attr = [topic_entity_weights[edge] for edge in unique_topic_entity_edges]
        subtopic_entity_edge_attr = [subtopic_entity_weights[edge] for edge in unique_subtopic_entity_edges]
        trigger_entity_edge_attr = [trigger_entity_weights[edge] for edge in unique_trigger_entity_edges]
        argument_entity_edge_attr = [argument_entity_weights[edge] for edge in unique_argument_entity_edges]

        data = HeteroData()
        # print(f"data['topic']: {data['topic']}")
        data['topic'].x = torch.arange(len(node_dict['topic']) + 2)
        data['subtopic'].x = torch.arange(len(node_dict['subtopic']) + 2)
        data['trigger'].x = torch.arange(len(node_dict['trigger']) + 2)
        data['argument'].x = torch.arange(len(node_dict['argument']) + 2)
        data['entity'].x = torch.arange(len(node_dict['entity']) + 2)

        data['entity', 'to', 'topic'].edge_index = torch.tensor(entity_topic_edge_index, dtype=torch.long)
        data['entity', 'to', 'subtopic'].edge_index = torch.tensor(entity_subtopic_edge_index, dtype=torch.long)
        data['entity', 'to', 'trigger'].edge_index = torch.tensor(entity_trigger_edge_index, dtype=torch.long)
        data['entity', 'to', 'argument'].edge_index = torch.tensor(entity_argument_edge_index, dtype=torch.long)
        data['entity', 'to', 'entity'].edge_index = torch.tensor(entity_entity_edge_index, dtype=torch.long)
        data['topic', 'to', 'entity'].edge_index = torch.tensor(topic_entity_edge_index, dtype=torch.long)
        data['subtopic', 'to', 'entity'].edge_index = torch.tensor(subtopic_entity_edge_index, dtype=torch.long)
        data['trigger', 'to', 'entity'].edge_index = torch.tensor(trigger_entity_edge_index, dtype=torch.long)
        data['argument', 'to', 'entity'].edge_index = torch.tensor(argument_entity_edge_index, dtype=torch.long)

        data['entity', 'to', 'topic'].edge_attr = torch.tensor(entity_topic_edge_attr, dtype=torch.long)
        data['entity', 'to', 'subtopic'].edge_attr = torch.tensor(entity_subtopic_edge_attr, dtype=torch.long)
        data['entity', 'to', 'trigger'].edge_attr = torch.tensor(entity_trigger_edge_attr, dtype=torch.long)
        data['entity', 'to', 'argument'].edge_attr = torch.tensor(entity_argument_edge_attr, dtype=torch.long)
        data['entity', 'to', 'entity'].edge_attr = torch.tensor(entity_entity_edge_attr, dtype=torch.long)
        data['topic', 'to', 'entity'].edge_attr = torch.tensor(topic_entity_edge_attr, dtype=torch.long)
        data['subtopic', 'to', 'entity'].edge_attr = torch.tensor(subtopic_entity_edge_attr, dtype=torch.long)
        data['trigger', 'to', 'entity'].edge_attr = torch.tensor(trigger_entity_edge_attr, dtype=torch.long)
        data['argument', 'to', 'entity'].edge_attr = torch.tensor(argument_entity_edge_attr, dtype=torch.long)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Hetero Graph Construction, \nHetero Graph Path: {target_path} \nGraph Info: {data}")







        # for i in range(len(news_edge_attr)):
        #     src_news = news_edge_index[0][i]
        #     dst_news = news_edge_index[1][i]
        #     # print(f"src: {src_news}, dst: {dst_news}")
        #     weight = news_edge_attr[i]
        #     src_news_info = hetero_graph_basic_dict[src_news]
        #     dst_news_info = hetero_graph_basic_dict[dst_news]
        #     # print(f"src_news_info: {hetero_graph_basic_dict[src_news]}")
        #     src_topic, src_subtopic, src_arguments, src_triggers, src_entities = (src_news_info[key] for key in ["topic", "subtopic", "argument", "trigger", "entity"])
        #     dst_topic, dst_subtopic, dst_arguments, dst_triggers, dst_entities = (dst_news_info[key] for key in ["topic", "subtopic", "argument", "trigger", "entity"])
        #     # print(f"src: {src_topic}, {src_subtopic}, {src_arguments}, {src_triggers}, {src_entities}")
        #     # print(f"dst: {dst_topic}, {dst_subtopic}, {dst_arguments}, {dst_triggers}, {dst_entities}")
        #
        #
        #     # topic -> topic
        #     # add_hetero_graph_edge(edge_weights, src_topic, 'topic', dst_topic, 'topic', 'topic_to_topic', weight)
        #
        #     # subtopic -> subtopic
        #     # add_hetero_graph_edge(edge_weights, src_subtopic, 'subtopic', dst_subtopic, 'subtopic', 'subtopic_to_subtopic', weight)
        #
        #     # argument <-> argument
        #     # for src_argument in src_arguments:
        #     #     for dst_argument in dst_arguments:
        #     #         add_hetero_graph_edge(edge_weights, src_argument, 'argument', dst_argument, 'argument', 'argument_to_argument', weight)
        #     #         add_hetero_graph_edge(edge_weights, dst_argument, 'argument', src_argument, 'argument', 'argument_to_argument', weight)
        #
        #     # src_topic -> dst_entities、 src_subtopic -> dst_entities
        #     for dst_entity in dst_entities:
        #         add_hetero_graph_edge(edge_weights, src_topic, 'topic', dst_entity, 'entity', 'topic_to_entity', weight)
        #         add_hetero_graph_edge(edge_weights, src_subtopic, 'subtopic', dst_entity, 'entity', 'subtopic_to_entity', weight)
        #
        #
        #     # src_topic -> dst_arguments、 src_subtopic -> dst_arguments
        #     # for dst_argument in dst_arguments:
        #     #     add_hetero_graph_edge(edge_weights, src_topic, 'topic', dst_argument, 'argument', 'topic_to_argument', weight)
        #     #     add_hetero_graph_edge(edge_weights, src_subtopic, 'subtopic', dst_argument, 'argument', 'subtopic_to_argument', weight)
        #
        #     # src_trigger -> dst_entities、 src_trigger -> dst_arguments
        #     for src_trigger in src_triggers:
        #         for dst_entity in dst_entities:
        #             add_hetero_graph_edge(edge_weights, src_trigger, 'trigger', dst_entity, 'entity', 'trigger_to_entity', weight)
        #         # for dst_argument in dst_arguments:
        #         #     add_hetero_graph_edge(edge_weights, src_trigger, 'trigger', dst_argument, 'argument', 'trigger_to_argument', weight)
        #     # src_argument -> dst_entities
        #     for src_argument in src_arguments:
        #         for dst_entity in dst_entities:
        #             add_hetero_graph_edge(edge_weights, src_argument, 'argument', dst_entity, 'entity', 'argument_to_entity', weight)
        #
        #     for src_entity in src_entities:
        #         # src_entities <-> dst_entities
        #         for dst_entity in dst_entities:
        #             add_hetero_graph_edge(edge_weights, src_entity, 'entity', dst_entity, 'entity', 'entity_to_entity', weight)
        #             # add_hetero_graph_edge(edge_weights, dst_entity, 'entity', src_entity, 'entity', 'entity_to_entity', weight)
        #
        #         # src_entities -> dst_topic
        #         add_hetero_graph_edge(edge_weights, src_entity, 'entity', dst_topic, 'topic', 'entity_to_topic', weight)
        #         # src_entities -> dst_subtopic
        #         add_hetero_graph_edge(edge_weights, src_entity, 'entity', dst_subtopic, 'subtopic', 'entity_to_subtopic', weight)
        #         # src_entities -> dst_triggers
        #         for dst_trigger in dst_triggers:
        #             add_hetero_graph_edge(edge_weights, src_entity, 'entity', dst_trigger, 'trigger', 'entity_to_trigger', weight)
        #         # src_entities -> dst_arguments
        #         for dst_argument in dst_arguments:
        #             add_hetero_graph_edge(edge_weights, src_entity, 'entity', dst_argument, 'argument', 'entity_to_argument', weight)
        #
        # update_hetero_graph_with_edges(hetero_graph, edge_weights)
        # torch.save(hetero_graph, target_path)
        # # print(hetero_graph)
        # print(f"[{mode}] Finish Hetero Graph Construction, \nHetero Graph Path: {target_path} \nGraph Info: {hetero_graph}")

    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path)
        data = HeteroData()
        # print(f"data['topic']: {data['topic']}")
        data['topic'].x = origin_graph['topic']
        data['subtopic'].x = origin_graph['subtopic']
        data['trigger'].x = origin_graph['trigger']
        data['argument'].x = origin_graph['argument']
        data['entity'].x = origin_graph['entity']

        data['entity', 'to', 'topic'].edge_index = origin_graph['entity', 'to', 'topic'].edge_index
        data['entity', 'to', 'subtopic'].edge_index = origin_graph['entity', 'to', 'subtopic'].edge_index
        data['entity', 'to', 'trigger'].edge_index = origin_graph['entity', 'to', 'trigger'].edge_index
        data['entity', 'to', 'argument'].edge_index = origin_graph['entity', 'to', 'argument'].edge_index
        data['entity', 'to', 'entity'].edge_index = origin_graph['entity', 'to', 'entity'].edge_index
        data['topic', 'to', 'entity'].edge_index = origin_graph['topic', 'to', 'entity'].edge_index
        data['subtopic', 'to', 'entity'].edge_index = origin_graph['subtopic', 'to', 'entity'].edge_index
        data['trigger', 'to', 'entity'].edge_index = origin_graph['trigger', 'to', 'entity'].edge_index
        data['argument', 'to', 'entity'].edge_index = origin_graph['argument', 'to', 'entity'].edge_index

        data['entity', 'to', 'topic'].edge_attr = origin_graph['entity', 'to', 'topic'].edge_attr
        data['entity', 'to', 'subtopic'].edge_attr = origin_graph['entity', 'to', 'subtopic'].edge_attr
        data['entity', 'to', 'trigger'].edge_attr = origin_graph['entity', 'to', 'trigger'].edge_attr
        data['entity', 'to', 'argument'].edge_attr = origin_graph['entity', 'to', 'argument'].edge_attr
        data['entity', 'to', 'entity'].edge_attr = origin_graph['entity', 'to', 'entity'].edge_attr
        data['topic', 'to', 'entity'].edge_attr = origin_graph['topic', 'to', 'entity'].edge_attr
        data['subtopic', 'to', 'entity'].edge_attr = origin_graph['subtopic', 'to', 'entity'].edge_attr
        data['trigger', 'to', 'entity'].edge_attr = origin_graph['trigger', 'to', 'entity'].edge_attr
        data['argument', 'to', 'entity'].edge_attr = origin_graph['argument', 'to', 'entity'].edge_attr

        torch.save(data, target_path)
        print(f"[{mode}] Finish Hetero Graph Construction, \nHetero Graph Path: {target_path} \nGraph Info: {data}")





def add_hetero_graph_edge(edge_weights, src, src_type, dst, dst_type, relation, weight=1):
    """
    添加边或更新边权重。
    """
    key = (src_type, relation, dst_type, src, dst)
    edge_weights[key] += weight
    # print(f"Add edge {src_type}_{dst_type}: {src}_{dst}, add weight = {weight}, now weight = {edge_weights[key]}")


def update_hetero_graph_with_edges(graph, edge_weights):
    """
    将边及其权重添加到异构图中。
    """
    # print(f"len: {len(edge_weights.items())}")
    for (src_type, relation, dst_type, src, dst), weight in edge_weights.items():
        if 'edge_index' not in graph[(src_type, dst_type)]:
            graph[(src_type, dst_type)]['edge_index'] = torch.tensor([], dtype=torch.long).reshape(2, 0)
            graph[(src_type, dst_type)]['edge_weight'] = torch.tensor([], dtype=torch.int)

        edge_index = graph[(src_type, dst_type)]['edge_index']
        edge_weight = graph[(src_type, dst_type)]['edge_weight']
        # print(f"edge_index: {edge_index}")
        # 更新边和权重
        graph[(src_type, dst_type)]['edge_index'] = torch.cat(
            [edge_index, torch.tensor([[src], [dst]], dtype=torch.long)], dim=1
        )
        graph[(src_type, dst_type)]['edge_weight'] = torch.cat(
            [edge_weight, torch.tensor([weight], dtype=torch.int)]
        )
    # print(f"edge_weight.items(): {edge_weights.items()}")
    # print(f"len: {len(edge_weights.items())}")

    # for (src_type, relation, dst_type, src, dst), weight in edge_weights.items():

    #     if f'{relation}_edge_index' not in graph[(src_type, dst_type)]:
    #         graph[(src_type, dst_type)][f'{relation}_edge_index'] = torch.tensor([], dtype=torch.long).reshape(2, 0)
    #         graph[(src_type, dst_type)][f'{relation}_edge_weight'] = torch.tensor([], dtype=torch.int)
    #
    #     edge_index = graph[(src_type, dst_type)][f'{relation}_edge_index']
    #     edge_weight = graph[(src_type, dst_type)][f'{relation}_edge_weight']
    #
    #     # 更新边和权重
    #     graph[(src_type, dst_type)][f'{relation}_edge_index'] = torch.cat(
    #         [edge_index, torch.tensor([[src], [dst]], dtype=torch.long)], dim=1
    #     )
    #     graph[(src_type, dst_type)][f'{relation}_edge_weight'] = torch.cat(
    #         [edge_weight, torch.tensor([weight], dtype=torch.int)]
    #     )


def extend_entity_neighbors_pool(cfg, mode):
    """
        扩展当前candidate_news的异构图邻居
    """
    print(f"[{mode}] Start to process hetero graph adjacent pool")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    hetero_neighbor_dict_path = Path(data_dir[mode]) / f"hetero_neighbor_dict.bin"
    hetero_weights_dict_path = Path(data_dir[mode]) / f"hetero_weights_dict.bin"
    hetero_graph_news_input_path = Path(data_dir[mode]) / f"hetero_graph_news_input.bin"

    hetero_graph_adjacent_pool_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_pool.bin"
    hetero_graph_adjacent_weights_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_weights.bin"

    # hetero_graph_adjacent_pool_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_pool.json"
    # hetero_graph_adjacent_weights_path = Path(data_dir[mode]) / f"hetero_graph_adjacent_weights.json"

    reprocess_flag = False
    for file_path in [hetero_neighbor_dict_path, hetero_weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Hetero Graph Neighbor dict exist !")
        return

    hetero_neighbors = pickle.load(open(hetero_neighbor_dict_path, 'rb'))
    hetero_neighbors_weights = pickle.load(open(hetero_weights_dict_path, 'rb'))
    hetero_graph_news_input = pickle.load(open(hetero_graph_news_input_path, 'rb'))


    # topic_id = hetero_graph_news_input[:, 0]
    # subtopic_id = hetero_graph_news_input[:, 1]
    # trigger_ids = hetero_graph_news_input[:, 2:5]
    # argument_ids = hetero_graph_news_input[:, 5:10]
    # entity_ids = hetero_graph_news_input[:, -5:]

    # adjacent_pool_types = {'strong_adjacent_pool', 'public_adjacent_pool', 'indirect_adjacent_pool'}
    # adjacent_poll_weights_types = {'strong_adjacent_weights', 'public_adjacent_weights', 'indirect_adjacent_weights'}
    hetero_graph_adjacent_pool = {}
    hetero_graph_adjacent_weights = {}

    for news_index, hetero_input in enumerate(hetero_graph_news_input):
        topic_id = hetero_input[0]
        subtopic_id = hetero_input[1]
        trigger_ids = hetero_input[2:5]
        argument_ids = hetero_input[5:10]
        entity_ids = hetero_input[-5:]

        strong_adjacent_pool, strong_adjacent_weights = build_strong_adjacent_pool(cfg, entity_ids, hetero_neighbors,
                                                                                   hetero_neighbors_weights)
        public_adjacent_pool, public_adjacent_weights = build_public_adjacent_pool(cfg, topic_id, subtopic_id,
                                                                                   trigger_ids, argument_ids,
                                                                                   hetero_neighbors,
                                                                                   hetero_neighbors_weights)
        indirect_entity_adjacent_pool, indirect_entity_adjacent_weights, indirect_non_entity_adjacent_pool, indirect_non_entity_adjacent_weights = build_indirect_adjacent_poll(cfg, entity_ids, strong_adjacent_pool, strong_adjacent_weights, hetero_neighbors, hetero_neighbors_weights)

        if news_index in hetero_graph_adjacent_pool:
            continue
        else:
            hetero_graph_adjacent_pool[news_index] = {}
            hetero_graph_adjacent_weights[news_index] = {}

        hetero_graph_adjacent_pool[news_index]['strong_adjacent_pool'] = strong_adjacent_pool
        hetero_graph_adjacent_pool[news_index]['public_adjacent_pool'] = public_adjacent_pool
        hetero_graph_adjacent_pool[news_index]['indirect_entity_adjacent_pool'] = indirect_entity_adjacent_pool
        hetero_graph_adjacent_pool[news_index]['indirect_non_entity_adjacent_pool'] = indirect_non_entity_adjacent_pool
        hetero_graph_adjacent_weights[news_index]['strong_adjacent_weights'] = strong_adjacent_weights
        hetero_graph_adjacent_weights[news_index]['public_adjacent_weights'] = public_adjacent_weights
        hetero_graph_adjacent_weights[news_index]['indirect_entity_adjacent_weights'] = indirect_entity_adjacent_weights
        hetero_graph_adjacent_weights[news_index]['indirect_non_entity_adjacent_weights'] = indirect_non_entity_adjacent_weights


    with open(hetero_graph_adjacent_pool_path, 'w', encoding='utf-8') as f:
        json.dump(hetero_graph_adjacent_pool, f, ensure_ascii=False, indent=4)
    with open(hetero_graph_adjacent_weights_path, 'w', encoding='utf-8') as f:
        json.dump(hetero_graph_adjacent_weights, f, ensure_ascii=False, indent=4)
    # pickle.dump(hetero_graph_adjacent_pool, open(hetero_graph_adjacent_pool_path, 'wb'))
    # pickle.dump(hetero_graph_adjacent_weights, open(hetero_graph_adjacent_weights_path, 'wb'))



def build_strong_adjacent_pool(cfg, entity_ids, hetero_neighbors, hetero_neighbors_weights):
    """
        从每个entity_id开始扩展entity邻居
        strong_adjacent_pool: []
        strong_adjacent_poll_weight: []

        return:
        strong_adjacent_pool: [entity_id's neighbors for entity_id in entity_ids]
        strong_adjacent_weights: [entity_id's neighbor's weight for entity_id in entity_ids]
    """
    strong_adjacent_pool = []
    strong_adjacent_weights = []
    entity_neighbors = hetero_neighbors['entity_entity']
    entity_neighbors_weights = hetero_neighbors_weights['entity_entity']
    # print(f"entity_neighbors: {entity_neighbors}")

    # 终点是entity_id的所有起始节点
    for entity_id in entity_ids:
        if entity_id == 0:
            continue
        cur_entity_neighbors = []
        cur_entity_neighbors_weights = []
        entity_neighbor = entity_neighbors[entity_id][:cfg.model.strong_entity_adjacent_num]
        entity_neighbor_weight = entity_neighbors_weights[entity_id][:cfg.model.strong_entity_adjacent_num]
        # entity_neighbor = entity_neighbors[entity_id][:]
        # entity_neighbor_weight = entity_neighbors_weights[entity_id][:]
        cur_entity_neighbors.extend(entity_neighbor)
        cur_entity_neighbors_weights.extend(entity_neighbor_weight)
        # if len(cur_entity_neighbors) < cfg.model.strong_entity_adjacent_num:
        #     for i in range(cfg.model.strong_entity_adjacent_num - len(cur_entity_neighbors)):
        #         cur_entity_neighbors.append(entity_id)
        #         cur_entity_neighbors_weights.append(1)
        strong_adjacent_pool.extend(cur_entity_neighbors)
        strong_adjacent_weights.extend(cur_entity_neighbors_weights)
        # print(f"entity_id: {entity_id}: \n strong_adjacent_pool: {cur_entity_neighbors} \nstrong_adjacent_weights: {cur_entity_neighbors_weights}")

    strong_adjacent_pool, strong_adjacent_weights = duplicate_elem_combined(strong_adjacent_pool, strong_adjacent_weights)
    return strong_adjacent_pool, strong_adjacent_weights



def build_public_adjacent_pool(cfg, topic_id, subtopic_id, trigger_ids, argument_ids, hetero_neighbors, hetero_neighbors_weights):
    """
        从topic_id、subtopic_id、和每个trigger_id和每个argument_id向外扩展entity邻居
        从每个项中各取三个entity邻居

    """
    public_adjacent_pool = []
    public_adjacent_weights = []

    topic_entity_neighbors = hetero_neighbors['topic_entity']
    topic_entity_neighbors_weights = hetero_neighbors_weights['topic_entity']
    subtopic_entity_neighbors = hetero_neighbors['subtopic_entity']
    subtopic_entity_neighbors_weights = hetero_neighbors_weights['subtopic_entity']
    trigger_entity_neighbors = hetero_neighbors['trigger_entity']
    trigger_entity_neighbors_weights = hetero_neighbors_weights['trigger_entity']
    argument_entity_neighbors = hetero_neighbors['argument_entity']
    argument_entity_neighbors_weights = hetero_neighbors_weights['argument_entity']


    cur_topic_entity_neighbors = topic_entity_neighbors[topic_id][:cfg.model.public_entity_adjacent_num]
    cur_topic_entity_neighbors_weights = topic_entity_neighbors_weights[topic_id][:cfg.model.public_entity_adjacent_num]
    # cur_topic_entity_neighbors = topic_entity_neighbors[topic_id][:]
    # cur_topic_entity_neighbors_weights = topic_entity_neighbors_weights[topic_id][:]
    # if len(cur_topic_entity_neighbors) < cfg.model.public_entity_adjacent_num:
    #     for i in range(cfg.model.public_entity_adjacent_num - len(cur_topic_entity_neighbors)):
    #         cur_topic_entity_neighbors.append(0)
    #         cur_topic_entity_neighbors_weights.append(1)
    cur_subtopic_entity_neighbors = subtopic_entity_neighbors[subtopic_id][:cfg.model.public_entity_adjacent_num]
    cur_subtopic_entity_neighbors_weights = subtopic_entity_neighbors_weights[subtopic_id][:cfg.model.public_entity_adjacent_num]
    # cur_subtopic_entity_neighbors = subtopic_entity_neighbors[subtopic_id][:]
    # cur_subtopic_entity_neighbors_weights = subtopic_entity_neighbors_weights[subtopic_id][:]
    # if len(cur_subtopic_entity_neighbors) < cfg.model.public_entity_adjacent_num:
    #     for i in range(cfg.model.public_entity_adjacent_num - len(cur_subtopic_entity_neighbors)):
    #         cur_subtopic_entity_neighbors.append(0)
    #         cur_subtopic_entity_neighbors_weights.append(1)
    cur_trigger_entity_neighbors = []
    cur_trigger_entity_neighbors_weights = []
    cur_argument_entity_neighbors = []
    cur_argument_entity_neighbors_weights = []

    for trigger_id in trigger_ids:
        if trigger_id == 0:
            continue
        # print(f"trigger_entity_neighbors[trigger_id][:cfg.model.public_entity_adjacent_num]: {trigger_entity_neighbors[trigger_id][:cfg.model.public_entity_adjacent_num]}")
        for i, entity_id in enumerate(trigger_entity_neighbors[trigger_id][:cfg.model.public_entity_adjacent_num]):
        # for i, entity_id in enumerate(trigger_entity_neighbors[trigger_id][:]):
            # cur_trigger_entity_neighbors.extend(trigger_entity_neighbors[trigger_id][:cfg.model.public_entity_adjacent_num])
            # cur_trigger_entity_neighbors_weights.extend(trigger_entity_neighbors_weights[trigger_id][:cfg.model.public_entity_adjacent_num])
            cur_trigger_entity_neighbors.append(entity_id)
            cur_trigger_entity_neighbors_weights.append(trigger_entity_neighbors_weights[trigger_id][:cfg.model.public_entity_adjacent_num][i])
            # cur_trigger_entity_neighbors_weights.append(trigger_entity_neighbors_weights[trigger_id][:][i])
    # if len(cur_trigger_entity_neighbors) < len(trigger_ids) * cfg.model.public_entity_adjacent_num:
    #     for i in range(len(trigger_ids) * cfg.model.public_entity_adjacent_num - len(cur_trigger_entity_neighbors)):
    #         cur_trigger_entity_neighbors.append(0)
    #         cur_trigger_entity_neighbors_weights.append(1)


    for argument_id in argument_ids:
        if argument_id == 0:
            continue
        # cur_argument_entity_neighbors.append(argument_entity_neighbors[argument_id][:cfg.model.public_entity_adjacent_num])
        # cur_argument_entity_neighbors_weights.append(argument_entity_neighbors_weights[argument_id][:cfg.model.public_entity_adjacent_num])
        for i, entity_id in enumerate(argument_entity_neighbors[argument_id][:cfg.model.public_entity_adjacent_num]):
        # for i, entity_id in enumerate(argument_entity_neighbors[argument_id][:]):
            cur_argument_entity_neighbors.append(entity_id)
            cur_argument_entity_neighbors_weights.append(argument_entity_neighbors_weights[argument_id][:cfg.model.public_entity_adjacent_num][i])
            # cur_argument_entity_neighbors_weights.append(argument_entity_neighbors_weights[argument_id][:][i])
    # if len(cur_argument_entity_neighbors) < len(argument_ids) * cfg.model.public_entity_adjacent_num:
    #     for i in range(len(argument_ids) * cfg.model.public_entity_adjacent_num - len(cur_argument_entity_neighbors)):
    #         cur_argument_entity_neighbors.append(0)
    #         cur_argument_entity_neighbors_weights.append(1)

    for neighbor in (cur_topic_entity_neighbors, cur_subtopic_entity_neighbors, cur_trigger_entity_neighbors, cur_argument_entity_neighbors):
        public_adjacent_pool.extend(neighbor)

    for weight in (cur_topic_entity_neighbors_weights, cur_subtopic_entity_neighbors_weights, cur_trigger_entity_neighbors_weights, cur_argument_entity_neighbors_weights):
        public_adjacent_weights.extend(weight)

    public_adjacent_pool, public_adjacent_weights = duplicate_elem_combined(public_adjacent_pool, public_adjacent_weights)
    # print(f"public_adjacent_pool: {public_adjacent_pool}")
    # print(f"public_adjacent_weights: {public_adjacent_weights}")
    return public_adjacent_pool, public_adjacent_weights




def build_indirect_adjacent_poll(cfg, entity_ids, strong_adjacent_poll, strong_adjacent_weights, hetero_neighbors, hetero_neighbors_weights):
    """
        1. 从strong_adjacent_poll中的entity_ids继续向外扩展entity邻居
        2. 先从每个entity_id开始向外延展topic_id、subtopic_id、trigger_ids、argument_ids，
           再向外扩展entity邻居
    """

    entity_entity_neighbors = hetero_neighbors['entity_entity']
    entity_topic_neighbors = hetero_neighbors['entity_topic']
    entity_subtopic_neighbors = hetero_neighbors['entity_subtopic']
    entity_trigger_neighbors = hetero_neighbors['entity_trigger']
    entity_argument_neighbors = hetero_neighbors['entity_argument']
    topic_entity_neighbors = hetero_neighbors['topic_entity']
    subtopic_entity_neighbors = hetero_neighbors['subtopic_entity']
    trigger_entity_neighbors = hetero_neighbors['trigger_entity']
    argument_entity_neighbors = hetero_neighbors['argument_entity']

    entity_entity_weights = hetero_neighbors_weights['entity_entity']
    entity_topic_weights = hetero_neighbors_weights['entity_topic']
    entity_subtopic_weights = hetero_neighbors_weights['entity_subtopic']
    entity_trigger_weights = hetero_neighbors_weights['entity_trigger']
    entity_argument_weights = hetero_neighbors_weights['entity_argument']
    topic_entity_weights = hetero_neighbors_weights['topic_entity']
    subtopic_entity_weights = hetero_neighbors_weights['subtopic_entity']
    trigger_entity_weights = hetero_neighbors_weights['trigger_entity']
    argument_entity_weights = hetero_neighbors_weights['argument_entity']

    indirect_entity_neighbors = []
    # indirect_topic_entity_neighbors = []
    # indirect_subtopic_entity_neighbors = []
    # indirect_trigger_entity_neighbors = []
    # indirect_argument_entity_neighbors = []
    indirect_non_entity_neighbors = []

    indirect_entity_weights = []
    # indirect_topic_entity_weights = []
    # indirect_subtopic_entity_weights = []
    # indirect_trigger_entity_weights = []
    # indirect_argument_weights = []
    indirect_non_entity_weights = []

    # def check_and_fill(neighbors, weights, target_len, neighbor_fill_elem, weight_fill_elem):
    #     if len(neighbors) < target_len:
    #         for i in range(target_len - len(neighbors)):
    #             neighbors.append(neighbor_fill_elem)
    #             weights.append(weight_fill_elem)
    #
    #     return neighbors, weights


    # 1. 从strong_adjacent_poll中的entity向外扩展邻居（entity二阶邻居)
    # print(f"strong_adjacent_poll: {strong_adjacent_poll}")
    # print(f"strong_adjacent_weight: {strong_adjacent_weights}")
    for i, direct_entity_id in enumerate(strong_adjacent_poll):
        # direct_entity_ids: entity_ids[i]的一阶entity邻居
        cur_weight = strong_adjacent_weights[i]
        # for direct_entity_id in direct_entity_ids:
        if direct_entity_id == 0:
            continue
        cur_neighbors = entity_entity_neighbors[direct_entity_id][:cfg.model.second_order_entity_neighbors_num]
        cur_weights = entity_entity_weights[direct_entity_id][:cfg.model.second_order_entity_neighbors_num]
        # cur_neighbors = entity_entity_neighbors[direct_entity_id][:]
        # cur_weights = entity_entity_weights[direct_entity_id][:]
        for weight in cur_weights:
            weight += cur_weight
        # cur_neighbors, cur_weights = check_and_fill(cur_neighbors, cur_weights, cfg.model.indirect_adjacent_entity_num, direct_entity_id, 1)
        indirect_entity_neighbors.extend(cur_neighbors)
        indirect_entity_weights.extend(cur_weights)


    # 2. 从每个entity扩展到其他其中结点类型的邻居结点，再从这些邻居结点出发扩展entity邻居
    for entity_id in entity_ids:
        if entity_id == 0:
            continue
        # entity->topic的一阶邻居
        entity_topic_first_order_neighbors = entity_topic_neighbors[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        entity_topic_first_order_weights = entity_topic_weights[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        # entity_topic_first_order_neighbors = entity_topic_neighbors[entity_id][:]
        # entity_topic_first_order_weights = entity_topic_weights[entity_id][:]
        # entity_topic_first_order_neighbors, entity_topic_first_order_weights = check_and_fill(entity_topic_first_order_neighbors, entity_topic_first_order_weights, cfg.model.indirect_adjacent_non_entity_num, 0, 1)
        # topic->entity
        for i, topic_id in enumerate(entity_topic_first_order_neighbors):
            if topic_id == 0:
                continue
            weight = entity_topic_first_order_weights[i]
            entity_topic_entity_second_order_neighbors = topic_entity_neighbors[topic_id][:cfg.model.indirect_adjacent_entity_num]
            entity_topic_entity_second_order_weights = topic_entity_weights[topic_id][:cfg.model.indirect_adjacent_entity_num]
            # entity_topic_entity_second_order_neighbors = topic_entity_neighbors[topic_id][:]
            # entity_topic_entity_second_order_weights = topic_entity_weights[topic_id][:]
            for topic_entity_weight in entity_topic_entity_second_order_weights:
                topic_entity_weight += weight
            # entity_topic_entity_second_order_neighbors, entity_topic_entity_second_order_weights = check_and_fill(entity_topic_entity_second_order_neighbors, entity_topic_entity_second_order_weights, cfg.model.indirect_adjacent_entity_num, entity_id, 1)
            indirect_non_entity_neighbors.extend(entity_topic_entity_second_order_neighbors)
            indirect_non_entity_weights.extend(entity_topic_entity_second_order_weights)


        # entity->subtopic的一阶邻居
        entity_subtopic_first_order_neighbors = entity_subtopic_neighbors[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        entity_subtopic_first_order_weights = entity_subtopic_weights[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        # entity_subtopic_first_order_neighbors = entity_subtopic_neighbors[entity_id][:]
        # entity_subtopic_first_order_weights = entity_subtopic_weights[entity_id][:]
        # entity_subtopic_first_order_neighbors, entity_subtopic_first_order_weights = check_and_fill(entity_subtopic_first_order_neighbors, entity_subtopic_first_order_weights, cfg.model.indirect_adjacent_non_entity_num, 0, 1)

        # subtopic->entity
        for i, subtopic_id in enumerate(entity_subtopic_first_order_neighbors):
            if subtopic_id == 0:
                continue
            weight = entity_subtopic_first_order_weights[i]
            entity_subtopic_entity_second_order_neighbors = subtopic_entity_neighbors[subtopic_id][:cfg.model.indirect_adjacent_entity_num]
            entity_subtopic_entity_second_order_weights = subtopic_entity_weights[subtopic_id][:cfg.model.indirect_adjacent_entity_num]
            # entity_subtopic_entity_second_order_neighbors = subtopic_entity_neighbors[subtopic_id][:]
            # entity_subtopic_entity_second_order_weights = subtopic_entity_weights[subtopic_id][:]
            for subtopic_entity_weight in entity_subtopic_entity_second_order_weights:
                subtopic_entity_weight += weight
            # entity_subtopic_entity_second_order_neighbors, entity_subtopic_entity_second_order_weights = check_and_fill(entity_subtopic_entity_second_order_neighbors, entity_subtopic_entity_second_order_weights, cfg.model.indirect_adjacent_entity_num, entity_id, 1)
            indirect_non_entity_neighbors.extend(entity_subtopic_entity_second_order_neighbors)
            indirect_non_entity_weights.extend(entity_subtopic_entity_second_order_weights)

        # entity->trigger的一阶邻居
        entity_trigger_first_order_neighbors = entity_trigger_neighbors[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        entity_trigger_first_order_weights = entity_trigger_weights[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        # entity_trigger_first_order_neighbors = entity_trigger_neighbors[entity_id][:]
        # entity_trigger_first_order_weights = entity_trigger_weights[entity_id][:]
        # entity_trigger_first_order_neighbors, entity_trigger_first_order_weights = check_and_fill(entity_trigger_first_order_neighbors, entity_trigger_first_order_weights, cfg.model.indirect_adjacent_non_entity_num, 0, 1)

        # trigger->entity
        for i, trigger_id in enumerate(entity_trigger_first_order_neighbors):
            if trigger_id == 0:
                continue
            weight = entity_trigger_first_order_weights[i]
            entity_trigger_entity_second_order_neighbors = trigger_entity_neighbors[trigger_id][:cfg.model.indirect_adjacent_entity_num]
            entity_trigger_entity_second_order_weights = trigger_entity_weights[trigger_id][:cfg.model.indirect_adjacent_entity_num]
            # entity_trigger_entity_second_order_neighbors = trigger_entity_neighbors[trigger_id][:]
            # entity_trigger_entity_second_order_weights = trigger_entity_weights[trigger_id][:]
            for trigger_entity_weight in entity_trigger_entity_second_order_weights:
                trigger_entity_weight += weight
            indirect_non_entity_neighbors.extend(entity_trigger_entity_second_order_neighbors)
            indirect_non_entity_weights.extend(entity_trigger_entity_second_order_weights)


        # entity->argument的一阶邻居
        entity_argument_first_order_neighbors = entity_argument_neighbors[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        entity_argument_first_order_weights = entity_argument_weights[entity_id][:cfg.model.indirect_adjacent_non_entity_num]
        # entity_argument_first_order_neighbors = entity_argument_neighbors[entity_id][:]
        # entity_argument_first_order_weights = entity_argument_weights[entity_id][:]
        # argument->entity
        for i, argument_id in enumerate(entity_argument_first_order_neighbors):
            if argument_id == 0:
                continue
            weight = entity_argument_first_order_weights[i]
            entity_argument_entity_second_order_neighbors = argument_entity_neighbors[argument_id][:cfg.model.indirect_adjacent_entity_num]
            entity_argument_entity_second_order_weights = argument_entity_weights[argument_id][:cfg.model.indirect_adjacent_entity_num]
            # entity_argument_entity_second_order_neighbors = argument_entity_neighbors[argument_id][:]
            # entity_argument_entity_second_order_weights = argument_entity_weights[argument_id][:]
            for argument_entity_weight in entity_argument_entity_second_order_weights:
                argument_entity_weight += weight
            indirect_non_entity_neighbors.extend(entity_argument_entity_second_order_neighbors)
            indirect_non_entity_weights.extend(entity_argument_entity_second_order_weights)

    # print(f"indirect_entity_neighbors: {indirect_entity_neighbors}")
    # print(f"indirect_entity_weights: {indirect_entity_weights}")
    indirect_entity_neighbors, indirect_entity_weights = duplicate_elem_combined(indirect_entity_neighbors, indirect_entity_weights)
    indirect_non_entity_neighbors, indirect_non_entity_weights = duplicate_elem_combined(indirect_non_entity_neighbors, indirect_non_entity_weights)
    return indirect_entity_neighbors, indirect_entity_weights, indirect_non_entity_neighbors, indirect_non_entity_weights


def duplicate_elem_combined(neighbors, weights):
    unique_dict = {}
    for node, weight in zip(neighbors, weights):
        if node in unique_dict:
            unique_dict[node] += weight
        else:
            unique_dict[node] = weight

    new_neighbors = list(unique_dict.keys())
    new_weights = list(unique_dict.values())

    return new_neighbors, new_weights


def prepare_subcategory_graph(cfg, mode='train'):
    print("Building subcategory graph...")
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    target_path = Path(data_dir[mode]) / "subcategory_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Subcategory graph exists!")
        return

    subcategory_dict = pickle.load(open(Path(data_dir[mode]) / "subcategory_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "subcategory_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        subcategory_indices = news_graph.x[:, -7:-6].numpy()
        print("subcategory_indices, ", subcategory_indices.shape)

        subcategory_edge_index = []
        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_subcategory = subcategory_indices[news_edge_src[i]]
            dest_subcategory = subcategory_indices[news_edge_dest[i]]

            src_subcategory_mask = src_subcategory > 0
            dest_subcategory_mask = dest_subcategory > 0
            src_subcategory = src_subcategory[src_subcategory_mask]
            dest_subcategory = dest_subcategory[dest_subcategory_mask]
            edges = list(itertools.product(src_subcategory, dest_subcategory)) * edge_weights[i]
            subcategory_edge_index.extend(edges)
        edge_weights = Counter(subcategory_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)


        # edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(subcategory_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(subcategory_dict) + 1)
        torch.save(data, target_path)
        print(f"[mode] Finish Subcategory Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(subcategory_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(subcategory_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Subcategory Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")


# def prepare_user_info(cfg):
#     print("Building user info...")



def prepare_preprocessed_data(cfg):
    # Entity vec process
    data_dir = {"train":cfg.dataset.train_dir, "val":cfg.dataset.val_dir, "test":cfg.dataset.test_dir}
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    train_combined_path = Path(data_dir['train']) / "combined_entity_embedding.vec"
    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {train_combined_path}")
    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    os.system("cat " + f"{train_entity_emb_path} {test_entity_emb_path}" + f" > {test_combined_path}")


    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")
    # prepare_distributed_data(cfg, "test")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    # prepare_preprocess_bin(cfg, "test")

    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')
    # prepare_news_graph(cfg, 'test')

    # prepare_event_graph(cfg, 'train')
    # prepare_event_graph(cfg, 'val')
    # prepare_event_graph(cfg, 'test')
    prepare_hetero_graph(cfg, "train")
    prepare_hetero_graph(cfg, "val")

    prepare_hetero_neighbor_list(cfg, "train")
    prepare_hetero_neighbor_list(cfg, 'val')

    extend_entity_neighbors_pool(cfg, 'train')
    extend_entity_neighbors_pool(cfg, 'val')

    build_direct_and_indirect_entity_pool(cfg, 'train')
    build_direct_and_indirect_entity_pool(cfg, 'val')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    # prepare_neighbor_list(cfg, 'test', 'news')

    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')
    # prepare_entity_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')
    # prepare_neighbor_list(cfg, 'test', 'entity')


    # prepare_neighbor_list(cfg, 'train', 'event')
    # prepare_neighbor_list(cfg, 'val', 'event')
    # prepare_neighbor_list(cfg, 'test', 'event')

    # prepare_abs_entity_graph(cfg, 'train')
    # prepare_abs_entity_graph(cfg, 'val')
    # prepare_abs_entity_graph(cfg, 'test')

    # prepare_neighbor_list(cfg, 'train', 'abs_entity')
    # prepare_neighbor_list(cfg, 'val', 'abs_entity')
    # prepare_neighbor_list(cfg, 'test', 'abs_entity')

    # prepare_subcategory_graph(cfg, 'train')
    # prepare_subcategory_graph(cfg, 'val')
    # prepare_subcategory_graph(cfg, 'test')

    # prepare_neighbor_list(cfg, 'train', 'subcategory')
    # prepare_neighbor_list(cfg, 'val', 'subcategory')
    # prepare_neighbor_list(cfg, 'test', 'subcategory')




    # # Entity vec process
    # data_dir = {"train":cfg.dataset.train_dir, "val":cfg.dataset.val_dir, "test":cfg.dataset.test_dir}
    # train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    # val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    # test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"
    #
    # train_combined_path = Path(data_dir['train']) / "combined_entity_embedding.vec"
    # val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    # test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"
    #
    # os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    # os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    # os.system("cat " + f"{train_entity_emb_path} {test_entity_emb_path}" + f" > {test_combined_path}")

    print("Finish prepare_preprocessed_data function.")

