"""
Common utils and tools.
"""
import pickle
import random

import pandas as pd
import torch
import numpy as np
import pyrootutils
from pathlib import Path
import torch.distributed as dist

import importlib
from omegaconf import DictConfig, ListConfig


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def load_model(cfg):
    framework = getattr(importlib.import_module(f"models.{cfg.model.model_name}"), cfg.model.model_name)

    if cfg.model.use_entity:
        entity_dict = pickle.load(open(Path(cfg.dataset.val_dir) / "entity_dict.bin", "rb"))
        entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"
        entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
    else:
        entity_emb = None
    
    # if cfg.model.use_abs_entity:
    #     abs_entity_dict = pickle.load(open(Path(cfg.dataset.val_dir) / "abs_entity_dict.bin", "rb"))
    #     abs_entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"
    #     abs_entity_emb = load_pretrain_emb(abs_entity_emb_path, abs_entity_dict, 100)
    # else:
    #     abs_entity_emb = None

    # if cfg.model.use_subcategory_graph:
    #     subcategory_dict = pickle.load(open(Path(cfg.dataset.val_dir) / "subcategory_dict.bin", "rb"))
    # else:
    #     subcategory_dict = None

    # if cfg.model.use_key_entity:
    #     key_entity_dict = pickle.load(open(Path(cfg.dataset.val_dir) / "key_entity_dict.bin", "rb"))
    #     key_entity_emb_path = Path(cfg.dataset)

    if cfg.dataset.dataset_lang == 'english':
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = load_pretrain_emb(cfg.path.glove_path, word_dict, cfg.model.word_emb_dim)
    else:
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = len(word_dict)
    model = framework(cfg, glove_emb=glove_emb, entity_emb=entity_emb, abs_entity_emb=None, subcategory_dict=None)

    return model


def save_model(cfg, model, optimizer=None, mark=None):
    file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{mark}.pth")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        },
        file_path)
    print(f"Model Saved. Path = {file_path}")


def load_pretrain_emb(embedding_file_path, target_dict, target_dim):
    embedding_matrix = np.zeros(shape=(len(target_dict) + 1, target_dim))
    have_item = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                # print(f"line = {line}")
                itme = line[0].decode()
                # print(f"target_dict: {target_dict}")
                if itme in target_dict:
                    # print(f"itme = {itme}")
                    index = target_dict[itme]
                    # print(f"itme = {itme}, index = {index}")
                    tp = [float(x) for x in line[1:]]
                    # print(f"tp = {tp}")
                    embedding_matrix[index] = np.array(tp)
                    have_item.append(itme)
    print('-----------------------------------------------------')
    print(f'Dict length: {len(target_dict)}')
    print(f'Have words: {len(have_item)}')
    miss_rate = (len(target_dict) - len(have_item)) / len(target_dict) if len(target_dict) != 0 else 0
    print(f'Missing rate: {miss_rate}')
    # print(f"[Common-load_pretrain_emb]: embedding_matrix = {embedding_matrix}")
    return embedding_matrix


def trans_to_nindex(nid, news_index):
    # return [news_index[i] if i in news_index else 0 for i in nids]
    return news_index[nid]
def trans_to_key_entity_nindex(nids, key_entity_index):
    return [key_entity_index[i] if i in key_entity_index else 0 for i in nids]


def load_key_entity_emb(cfg, mode, target_dim, key_entities, news_dict):
    # key_entity_input: news_id + [entity_id]
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # Path(data_dir[mode]) / "news_dict.bin", "rb")
    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    entity_emb_path = Path(data_dir[mode]) / "combined_entity_embedding.vec"
    entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
    # print(f"[Common-load_key_entity]: entity_emb = {entity_emb}") # OK
    # print(f"[Common-load_key_entity]: entity_emb.shape = {entity_emb.shape}") # [14991, 100]
    key_entity_emb = np.zeros(shape=(len(key_entities)+1, cfg.model.key_entity_size, target_dim))
    key_entity_emb_mask = np.zeros(shape=(len(key_entities)+1, cfg.model.key_entity_size))
    # print(f"[Common-load_key_entity]: len(key_entities) = {len(key_entities)}") # [51282]
    # print(f"[Common-load_key_entity]: key_entity_emb_init.shape = {key_entity_emb.shape}") # [51283, 8, 100]
    # print(f"[Common-load_key_entity_mask]: key_entity_emb_mask_init.shape = {key_entity_emb_mask.shape}")
    # key_entities is a dict, with news_id as the key, and [key_entity, key_entity_mask] as the value
    # key_entity_list[0] is key_entity_id, and key_entity_list[1] is key_entity_mask
    for news_id, key_entity_list in key_entities.items():
        news_idx = trans_to_nindex(news_id, news_dict)
        # print(f"[Common-load_key_entity]: news_id = {news_id}, news_idx = {news_idx}")
        idx = 0
        key_entity_ids, key_entity_mask = key_entity_list
        # print(f"[Common-load_key_entity]: key_entity_ids = {key_entity_ids}")
        # print(f"[Common-load_key_entity]: key_entity_mask = {key_entity_mask}")
        for key_entity_id in key_entity_ids:
            if key_entity_id == 0:
                key_entity_emb[news_idx, idx] = np.zeros(target_dim)
            else:
                key_entity_emb[news_idx, idx] = entity_emb[key_entity_id]
            idx += 1
        key_entity_emb_mask[news_idx] = key_entity_mask
    # print(f"[Common-load_key_entity_emb]: key_entity_emb = {key_entity_emb}")
    # for news_id, entity_ids in key_entities[0].items():
    #     # 拿到news_id在新闻字典中的序号
    #     news_idx = trans_to_nindex(news_id, news_dict)
    #     idx = 0
    #     for entity_id in entity_ids:
    #         if entity_id == 0:
    #             key_entity_emb[news_idx, idx] = np.zeros(target_dim)
    #         else:
    #             key_entity_emb[news_idx, idx] = entity_emb[entity_id]
    #         idx += 1
    # print(f"in load_key_entity_emb, ")
    # print(f"key_entity_emb.shape: {key_entity_emb.shape}")
    # print(f"key_entity_emb_mask.shape: {key_entity_emb_mask.shape}")
    # print(f"key_entity_emb.dtype: {key_entity_emb.dtype}")
    # print(f"key_entity_emb_mask.dtype: {key_entity_emb_mask.dtype}")
    return key_entity_emb, key_entity_emb_mask





def reduce_mean(result, nprocs):
    rt = result.detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key)+ '\t' + str(value))


def get_root():
    return pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "README.md"],
        pythonpath=True,
        dotenv=True,
    )


class EarlyStopping:
    """
    Early Stopping class
    """

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = 0.0

    def __call__(self, score):
        """
        The greater score, the better result. Be careful the symbol.
        """
        if score > self.best_score:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_score = score
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better
