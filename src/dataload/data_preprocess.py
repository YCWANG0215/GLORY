import collections
import os
import subprocess
import tempfile
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch import scatter
from torch_geometric.data import Data
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

# from models.oneie.predict import *


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
        key_entity_dict = pickle.load(open(Path(data_dir["train"]) / "key_entity_dict.bin", "rb"))
        # role_dict = pickle.load(open(Path(data_dir["train"]) / "role_dict.bin", "rb"))
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
        key_entity_dict = {}
        # role_dict = {}

    # category_dict = {}
    # subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.
    # abs_word_cnt = Counter()

    num_line = len(open(file_path, encoding='utf-8').readlines())

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}]Processing raw news"):
            # split one line
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, abs_entity_str = split_line
            update_dict(target_dict=news_dict, key=news_id)
            update_dict(target_dict=event_dict, key=news_id)
            provided_entity = []
            # Entity
            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
                [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
                provided_entity = [obj["SurfaceForms"][0] for obj in json.loads(t_entity_str) if obj["SurfaceForms"]]
            else:
                entity_ids = t_entity_str


            # Abstract Entity
            if abs_entity_str:
                abs_entity_ids = [obj["WikidataId"] for obj in json.loads(abs_entity_str)]
                [update_dict(target_dict=abs_entity_dict, key=abs_entity_id) for abs_entity_id in abs_entity_ids]
            else:
                abs_entity_ids = abs_entity_str

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
                    entity_text = " ".join(event_json["tokens"][start:end])
                    event_entities.append(entity_text)

                triggers_origin = event_json["graph"]["triggers"]
                triggers = []
                event_types = []
                event_type_confidence = []
                for trigger in triggers_origin:
                    start, end, _event_type, confidence = trigger
                    trigger_text = " ".join(event_json["tokens"][start:end])
                    triggers.append(trigger_text)
                    event_types.append(_event_type)
                    event_type_confidence.append(confidence)

                # roles_origin = event_json["graph"]["roles"]
                # roles = []
                # for role in roles_origin:
                #     # role: 事件类型索引、实体索引、角色类型、置信度
                #     event_type_idx, entity_idx, role_type, confidence = role
                #     roles.append(role_type)
                #     update_dict(target_dict=role_dict, key=role_type)

            # TODO 根据置信度确定最终事件类型
            if len(event_types) != 0:
                max_confidence = max(event_type_confidence)
                max_confidence_indices = [i for i, confidence in enumerate(event_type_confidence) if confidence == max_confidence]
                event_type = event_types[max_confidence_indices[0]]
            else:
                event_type = f"Regular.{category}"
            # print(f"event_type: {event_type}")
            update_dict(target_dict=event_type_dict, key=event_type)




            # provided_entity_str = [' '.join(entity) for entity in provided_entity]
            # provided_entity_tuples = [tuple(entity) for entity in provided_entity]
            # event_entities_tuples = [tuple(entity) for entity in event_entities]
            # key_entity_tuples = list(set(provided_entity_tuples) & set(event_entities_tuples))
            key_entity_list = list(set(provided_entity) & set(event_entities))
            key_entity = ' '.join(key_entity_list)
            key_entity_tokens = word_tokenize(key_entity.lower(), language=cfg.dataset.dataset_lang)
            # key_entities = list(set(provided_entity) & set(event_entities))
            # key_entity = [list(entity) for entity in key_entity_tuples]
            # key_entity = [list(entity) for entity in key_entities]
            # print(f"provided_entity: {provided_entity}")
            # print(f"event_entities: {event_entities}")
            # print(f"key_entity: {key_entity}")
            # print()
            update_dict(target_dict=key_entity_dict, key=news_id, value=key_entity_tokens)


            tokens = word_tokenize(title.lower(), language=cfg.dataset.dataset_lang)
            # abs_tokens = word_tokenize(abstract.lower(), language=cfg.dataset.dataset_lang)

            # TODO key_entity要不要Glove编码
            event = {
                "event_type": event_type,
                "event_type_idx": event_type_dict[event_type],
                "event_entities": event_entities,
                # "entity_role": roles,
                "category": category,
                "subcategory": subcategory,
                "triggers": triggers
            }
            # print(f"event: {event}")
            update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                                news_dict[news_id], abs_entity_ids])


            update_dict(target_dict=category_dict, key=category)
            update_dict(target_dict=subcategory_dict, key=subcategory)
            update_dict(target_dict=events, key=news_id, value=[event_type, event_type_dict[event_type], event_entities, category, subcategory, triggers])
            # update_dict(target_dict=events, key=news_id, value=event)

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

        # TODO 调用read_raw_news接受返回值变更
        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num]
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            # return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
            return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict, abs_entity_dict, event_type_dict, events, event_dict, key_entity_dict
        else:  # val, test
            # TODO 非训练集要不要返回category_dict、subcategory_dict
            # return news, news_dict, None, None, entity_dict, None
            return news, news_dict, category_dict, subcategory_dict, entity_dict, None, abs_entity_dict, event_type_dict, events, event_dict, key_entity_dict


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


def read_news_events(cfg, news, news_dict, event_type_dict=None, events=None, word_dict=None, category_dict=None, subcategory_dict=None, event_dict=None):
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
    return event_type, event_entity, triggers, category, subcategory

def prepare_preprocess_bin(cfg, mode):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if cfg.reprocess is True:
        # Glove
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict, abs_entity_dict, event_type_dict, events, event_dict, key_entity_dict = read_raw_news(
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
        pickle.dump(key_entity_dict, open(Path(data_dir[mode])/ "key_entity_dict.bin", "wb"))
        # pickle.dump(role_dict, open(Path(data_dir[mode]) / "role_dict.bin", "wb"))

        # nltk_news_features: news_title, news_entity, news_category, news_subcategory, news_index
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict, abs_entity_dict)
        # def read_news_events(cfg, news, news_dict, event_dict=None, events=None, word_dict=None, role_dict=None, category_dict=None, subcategory_dict=None,):
        nltk_event_features = read_news_events(cfg, nltk_news, nltk_news_dict, event_type_dict, events, word_dict, category_dict, subcategory_dict, event_dict)
        # news_input: 把输入的新闻特征信息连成一个大矩阵
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb"))
        event_input = np.concatenate([x for x in nltk_event_features], axis=1)
        pickle.dump(event_input, open(Path(data_dir[mode]) / "nltk_token_event.bin", "wb"))
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
        print(data)
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


def prepare_event_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    nltk_target_path = Path(data_dir[mode]) / "nltk_event_graph.pt"

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return


    # -----------------------------------------News Graph------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    origin_graph_path = Path(data_dir['train']) / "nltk_event_graph.pt"

    # news_dict: 新闻id+序号
    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    # event_dict: 事件类型id+序号
    event_dict = pickle.load(open(Path(data_dir[mode]) / "event_dict.bin", "rb"))
    # events: key->news_id, value: event
    events = pickle.load(open(Path(data_dir[mode]) / "events.bin", "rb"))

    # nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    nltk_token_event = pickle.load(open(Path(data_dir[mode])/ "nltk_token_event.bin", "rb"))

    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        # 读取behavior行数
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to Event Graph"):
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
                    # long_edge = [news_dict[news_id] for news_id in history]
                    # TODO long_edge存news_id?
                    long_edge = [event_dict[news_id] for news_id in history]
                    edge_list.append(long_edge)

        # edge count
        node_feat = nltk_token_event
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing event edge list"):
            # Trajectory Graph 轨迹图
            if cfg.model.use_graph_type == 0:
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
                    # TODO 生成双向边（无向图？）
                    # short_edges.append((edge[i + 1], edge[i]))
            elif cfg.model.use_graph_type == 1:
                # Co-occurence Graph 共现图
                for i in range(len(edge) - 1):
                    for j in range(i + 1, len(edge)):
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
        print(data)
        print(f"[{mode}] Finish Event Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")

    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_event

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)

        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk Event Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")




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

    # TODO 处理abs_entity_graph.pt、subcategory_graph.pt
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

    # TODO 有向图or无向图
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
    
    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print(f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")

# TODO 实体图
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
        print("news_graph,", news_graph)
        print(f'news_graph.x.shape: {news_graph.x.shape}')
        # entity_indices：每个新闻中包含的实体的索引
        # TODO 多了5维abs_entity
        entity_indices = news_graph.x[:, -13:-8].numpy()
        # entity_indices = news_graph.x[:, -8:-3].numpy()
        print("entity_indices, ", entity_indices.shape)
        # TODO what is entity_indices

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
            # TODO 不需要过滤无效subcategory索引？
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

        # TODO 是否无向图
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

def prepare_preprocessed_data(cfg):
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    prepare_preprocess_bin(cfg, "test")

    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')
    prepare_news_graph(cfg, 'test')

    # prepare_event_graph(cfg, 'train')
    # prepare_event_graph(cfg, 'val')
    # prepare_event_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    prepare_neighbor_list(cfg, 'test', 'news')

    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')
    prepare_entity_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')
    prepare_neighbor_list(cfg, 'test', 'entity')

    # TODO 新增START
    # prepare_neighbor_list(cfg, 'train', 'event')
    # prepare_neighbor_list(cfg, 'val', 'event')
    # prepare_neighbor_list(cfg, 'test', 'event')

    prepare_abs_entity_graph(cfg, 'train')
    prepare_abs_entity_graph(cfg, 'val')
    prepare_abs_entity_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'abs_entity')
    prepare_neighbor_list(cfg, 'val', 'abs_entity')
    prepare_neighbor_list(cfg, 'test', 'abs_entity')

    prepare_subcategory_graph(cfg, 'train')
    prepare_subcategory_graph(cfg, 'val')
    prepare_subcategory_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'subcategory')
    prepare_neighbor_list(cfg, 'val', 'subcategory')
    prepare_neighbor_list(cfg, 'test', 'subcategory')

    # TODO 新增END


    # Entity vec process
    data_dir = {"train":cfg.dataset.train_dir, "val":cfg.dataset.val_dir, "test":cfg.dataset.test_dir}
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    os.system("cat " + f"{train_entity_emb_path} {test_entity_emb_path}" + f" > {test_combined_path}")

    print("Finish prepare_preprocessed_data function.")
