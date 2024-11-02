import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle

from dataload.dataset import *

from utils.common import load_pretrain_emb

from utils.common import load_key_entity_emb


def load_data(cfg, mode='train', model=None, local_rank=0):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    # ------------- load news.tsv-------------
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    # print(f"len(news_dict): {len(news_index)}")
    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    # print(f"news_input.shape: {news_input.shape}")
    event_index = pickle.load(open(Path(data_dir[mode]) / "event_dict.bin", "rb"))
    # print(f"len(event_dict) = {len(event_index)}")
    event_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_event.bin", "rb"))
    event_type_dict = pickle.load(open(Path(data_dir[mode]) / "event_type_dict.bin", "rb"))
    # print(f"event_type_dict.len: {len(event_type_dict)}")
    # print(f"event_input.shape = {event_input.shape}")
    # key_entity_dict = pickle.load(open(Path(data_dir[mode]) / "key_entity_dict.bin", "rb"))
    key_entities = pickle.load(open(Path(data_dir[mode]) / "key_entities.bin", "rb"))
    # print(f"len(key_entities) = {len(key_entities)}")
    # key_entity_input_mask = pickle.load(open(Path(data_dir[mode]) / "key_entities_mask.bin", "rb"))

    entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"

    # key_entity_emb = None
    # config = cfg
    # print(f"news_input.shape = {news_input.shape}")
    # ------------- load behaviors_np{X}.tsv --------------
    if mode == 'train':
        key_entity_input, key_entity_input_mask = load_key_entity_emb(cfg, mode, 100, key_entities, news_index)
        # print(f"[Data_load-load_data]: key_entity_input: {key_entity_input}")
        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        # target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv"
        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")


            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            # if cfg.model.use_event:
            #     event_graph = torch.load(Path(data_dir[mode]) / "nltk_event_graph.pt")
            #     if cfg.model.directed is False:
            #         event_graph.edge_index, event_graph.edge_attr = to_undirected(event_graph.edge_index, event_graph.edge_attr)
            #     print(f"[{mode}] Event Graph Info: {event_graph}")

            if cfg.model.use_entity:
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            # TODO 添加摘要实体图、类别图 START
            if cfg.model.use_abs_entity:
                abs_entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "abs_entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in abs_entity_neighbors.values())
                print(f"[{mode}] abs_entity_neighbor list Length: {total_length}")
            else:
                abs_entity_neighbors = None

            if cfg.model.use_subcategory_graph:
                subcategory_neighbors = pickle.load(open(Path(data_dir[mode]) / "subcategory_neighbor_dict.bin", "rb"))
                # print(f"subcategory_neighbors: {subcategory_neighbors}")
                total_length = sum(len(lst) for lst in subcategory_neighbors.values())
                print(f"[{mode}] subcategory_neighbor list Length: {total_length}")
            else:
                subcategory_neighbors = None

            # TODO 添加摘要实体图、类别图 END


            # dataset = TrainGraphDataset(
            #     filename=target_file,
            #     news_index=news_index,
            #     news_input=news_input,
            #     local_rank=local_rank,
            #     cfg=cfg,
            #     neighbor_dict=news_neighbors_dict,
            #     news_graph=news_graph,
            #     entity_neighbors=entity_neighbors
            # )
            # TODO dataset()改动
            dataset = TrainGraphDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors,
                abs_entity_neighbors=abs_entity_neighbors,
                subcategory_neighbors=subcategory_neighbors,
                event_index=event_index,
                event_input=event_input,
                # key_entity_index=key_entity_dict,
                key_entity_input=key_entity_input,
                key_entity_input_mask = key_entity_input_mask
            )
            dataloader = DataLoader(dataset, batch_size=None)
            
        else:
            dataset = TrainDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                event_index=event_index,
                event_input=event_input
            )

            dataloader = DataLoader(dataset,
                                    batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    # batch_size=int(cfg.batch_size / 1),
                                    pin_memory=True)
        return dataloader
    elif mode in ['val', 'test']:
        # convert the news to embeddings
        key_entity_input, key_entity_input_mask = load_key_entity_emb(cfg, mode, 100, key_entities, news_index)
        news_dataset = NewsDataset(news_input)
        news_dataloader = DataLoader(news_dataset,
                                     batch_size=int(cfg.batch_size * cfg.gpu_num),
                                     # batch_size=int(cfg.batch_size / 1),
                                     num_workers=cfg.num_workers)
        # print(f"EventDataset event_input shape: {event_input.shape}")
        event_dataset = EventDataset(event_input)
        # print(f"event_dataset.shape: {event_dataset.shape}")
        event_dataloader = DataLoader(event_dataset,
                                      batch_size=int(cfg.batch_size * cfg.gpu_num),
                                      # batch_size=int(cfg.batch_size / 1),
                                      num_workers=cfg.num_workers)
        stacked_news = []
        with torch.no_grad():
            for news_batch in tqdm(news_dataloader, desc=f"[{local_rank}] Processing validation News Embedding"):
                # print(f"news_batch.shape: {news_batch.shape}") # [32, 43]
                # print(f"news_batch: {news_batch}")
                if cfg.model.use_graph:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                else:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                    # batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                # print(f"batch_emb.shape: {batch_emb.shape}") # [32, 400]
                stacked_news.append(batch_emb)
        # print(f"stacked_news shape: {len(stacked_news)}") # 2039
        news_emb = torch.cat(stacked_news, dim=0).cpu().numpy()
        # print(f"news_emb.shape: {news_emb.shape}") # [65239, 400]

        stacked_events = []
        with torch.no_grad():
            # local_rank = 'cpu'
            # idx = 1
            for event_batch in tqdm(event_dataloader, desc=f"[{local_rank}] Processing validation Event Embedding"):
                # print(f"event_batch.shape: {event_batch.shape}") # [32, 11]
                # print(f"event_batch: {event_batch}")
                # event_emb = model.module.event_encoder(event_batch.long().unsqueeze(0).to(local_rank), None).squeeze(0).detach()
                # TODO detach()
                # print(f"event idx{idx} join.")
                # event_emb = event_batch.long().unsqueeze(0).to(local_rank)
                # event_emb = event_batch.to(local_rank)
                # print(f"finish 1")
                # event_emb = model.module.event_encoder(event_emb, None)
                # print("finish 2")
                # event_emb = event_emb.squeeze(0)
                # print(f"finish 3")
                event_emb = model.module.event_encoder(event_batch.long().unsqueeze(0).to(local_rank), None).squeeze(0).detach()
                # print(f"event idx{idx} event_emb finish.")
                # idx += 1
                # event_emb = model.module.event_encoder(event_batch.long().to(local_rank), None)
                # print(f"event_emb.shape: {event_emb.shape}") # [32, 400]
                # print(f"event_emb: {event_emb}")
                stacked_events.append(event_emb)
                # print(f"stacked_events: {stacked_events}")


        events_emb = torch.cat(stacked_events, dim=0).cpu().numpy()
        # print(f"events_emb: {events_emb}")

        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))


            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            # if cfg.model.use_event:
            #     # event_graph = torch.load(Path(data_dir[mode]) / "nltk_event_graph.pt")
            #     # event_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "event_neighbor_dict.bin", "rb"))
            #     if cfg.model.directed is False:
            #         event_graph.edge_index, event_graph.edge_attr = to_undirected(event_graph.edge_index,
            #                                                                     event_graph.edge_attr)
            #     print(f"[{mode}] News Graph Info: {event_graph}")
            if cfg.model.use_entity:
                # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            if cfg.model.use_abs_entity:
                abs_entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "abs_entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in abs_entity_neighbors.values())
                print(f"[{mode}] abs_entity_neighbor list Length: {total_length}")
            else:
                abs_entity_neighbors = None

            if cfg.model.use_subcategory_graph:
                subcategory_neighbors = pickle.load(open(Path(data_dir[mode]) / "subcategory_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in subcategory_neighbors.values())
                print(f"[{mode}] subcategory_neighbor list Length: {total_length}")
            else:
                subcategory_neighbors = None

            if mode == 'val':
                dataset = ValidGraphDataset(
                    # filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
                    filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    # TODO change
                    news_entity=news_input[:,-13:-8],
                    # news_entity=news_input[:,-8:-3],
                    entity_neighbors=entity_neighbors,
                    abs_entity_neighbors=abs_entity_neighbors,
                    news_abs_entity=news_input[:, -5:],
                    subcategory_neighbors=subcategory_neighbors,
                    news_subcategory=news_input[:, -7:-6],
                    event_index=event_index,
                    event_input=events_emb,
                    # key_entity_index=key_entity_dict,
                    key_entity_input=key_entity_input,
                    key_entity_input_mask=key_entity_input_mask,
                )
            elif mode == 'test':
                dataset = ValidGraphDataset(
                    # filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    # TODO change
                    news_entity=news_input[:, -13:-8],
                    # news_entity=news_input[:,-8:-3],
                    entity_neighbors=entity_neighbors,
                    abs_entity_neighbors=abs_entity_neighbors,
                    news_abs_entity=news_input[:, -5:],
                    subcategory_neighbors=subcategory_neighbors,
                    news_subcategory=news_input[:, -7:-6],
                    event_index=event_index,
                    event_input=events_emb,
                    # key_entity_index=key_entity_dict,
                    key_entity_input=key_entity_input,
                    key_entity_input_mask=key_entity_input_mask
                )

            dataloader = DataLoader(dataset, batch_size=None)
            # dataloader = DataLoader(dataset, batch_size=None, num_workers=cfg.num_workers)

        else:
            if mode == 'val':
                dataset = ValidDataset(
                    # filename=Path(data_dir[mode]) / f"behaviors_0.tsv",
                    filename=Path(data_dir[mode]) / f"behaviors_{local_rank}.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    event_index=event_index,
                    events_emb=events_emb,
                )
            else:
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    event_index=event_index,
                    events_emb=events_emb,
                )

            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    # batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    # pin_memory=True, # collate_fn already puts data to GPU
                                    # num_workers=cfg.num_workers,
                                    collate_fn=lambda b: collate_fn(b, local_rank))
        return dataloader


def collate_fn(tuple_list, local_rank):
    clicked_news = [x[0] for x in tuple_list]
    clicked_mask = [x[1] for x in tuple_list]
    candidate_news = [x[2] for x in tuple_list]
    clicked_index = [x[3] for x in tuple_list]
    candidate_index = [x[4] for x in tuple_list]

    if len(tuple_list[0]) == 6:
        labels = [x[5] for x in tuple_list]
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index, labels
    else:
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index
