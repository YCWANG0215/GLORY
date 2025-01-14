import os.path
from pathlib import Path

import hydra
import numpy
import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from dataload.data_preprocess import *
from utils.metrics import *
from utils.common import *

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = "d5041d20fdb1117dc20f9cb129e7749bd549f8b9"
os.environ["WANDB_MODE"] = "offline"


def train(model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping):
    model.train()
    torch.set_grad_enabled(True)

    sum_loss = torch.zeros(1).to(local_rank)
    sum_auc = torch.zeros(1).to(local_rank)

    # torch.autograd.set_detect_anomaly(True)

    # TODO Add event extraction model
    for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels, clicked_event, candidate_event,  clicked_topic_ids, clicked_topic_ids_mask, clicked_subtopic_ids, clicked_subtopic_ids_mask, clicked_subtopic_news_ids, clicked_subtopic_news_ids_mask, clicked_event_mask,  indirect_entity_neighbors, indirect_entity_neighbors_mask) \
            in enumerate(tqdm(dataloader,
                              total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)),
                              desc=f"[{local_rank}] Training"), start=1):
        # print(f"[main.train]: clicked_key_entity = {clicked_key_entity}")
        subgraph = subgraph.to(local_rank, non_blocking=True)
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        # print(f"in main.py, type(candidate_entity = {type(candidate_entity)})")
        candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
        # print(f"candidate_entity: {candidate_entity}")
        # print(f"candidate_entity.type: {candidate_entity.type}")
        # print(f"candidate_entity.shape = {candidate_entity.shape}")
        entity_mask = entity_mask.to(local_rank, non_blocking=True)
        # print(f"candidate_entity_mask.type: {entity_mask.type}")
        # print(f"candidate_entity_mask.shape: {entity_mask.shape}")
        clicked_event = clicked_event.to(local_rank, non_blocking=True)
        candidate_event = candidate_event.to(local_rank, non_blocking=True)
        clicked_event_mask = clicked_event_mask.to(local_rank, non_blocking=True)
        # if len(clicked_key_entity) != 0:
        # clicked_key_entity = clicked_key_entity.to(local_rank, non_blocking=True)
        # print(f"clicked_key_entity: {clicked_key_entity}")
        # clicked_key_entity_mask = clicked_key_entity_mask.to(local_rank, non_blocking=True)
        # if len(candidate_key_entity) != 0:
        # candidate_key_entity = candidate_key_entity.to(local_rank, non_blocking=True)
        # candidate_key_entity_mask = candidate_key_entity_mask.to(local_rank, non_blocking=True)
        # if len(abs_candidate_entity) != 0:

        # abs_candidate_entity = abs_candidate_entity.to(local_rank, non_blocking=True)
        # abs_entity_mask = abs_entity_mask.to(local_rank, non_blocking=True)
        # print(f"in main.py, type(subcategories) = {type(subcategories)}")
        # print(f"in main.py, subcategories: {subcategories}")
        # subcategories = torch.tensor(subcategories)
        # print(f"in main.py, type(subcategories) = {type(subcategories)}")
        # print(f"in main.py, subcategories: {subcategories}")
        # if len(subcategories) != 0:
        # subcategories = subcategories.to(local_rank, non_blocking=True)
        clicked_topic_ids = clicked_topic_ids.to(local_rank, non_blocking=True)
        clicked_topic_ids_mask = clicked_topic_ids_mask.to(local_rank, non_blocking=True)
        # clicked_topic_news_ids = clicked_topic_news_ids.to(local_rank, non_blocking=True)
        # clicked_topic_news_ids_mask = clicked_topic_news_ids_mask.to(local_rank, non_blocking=True)
        clicked_subtopic_ids = clicked_subtopic_ids.to(local_rank, non_blocking=True)
        clicked_subtopic_ids_mask = clicked_subtopic_ids_mask.to(local_rank, non_blocking=True)
        # print(f"clicked_subtopic_news_ids.shape: {clicked_subtopic_news_ids.shape}")
        clicked_subtopic_news_ids = clicked_subtopic_news_ids.to(local_rank, non_blocking=True)
        # print(f"[train] clicked_subtopic_news_ids.shape: f{clicked_subtopic_news_ids.shape}")

        clicked_subtopic_news_ids_mask = clicked_subtopic_news_ids_mask.to(local_rank, non_blocking=True)

        # candidate_second_order_entity_neighbor = candidate_second_order_entity_neighbor.to(local_rank, non_blocking=True)
        # second_order_entity_mask = second_order_entity_mask.to(local_rank, non_blocking=True)
        indirect_entity_neighbors = indirect_entity_neighbors.to(local_rank, non_blocking=True)
        indirect_entity_neighbors_mask = indirect_entity_neighbors_mask.to(local_rank, non_blocking=True)
        # user_id = user_id.to(local_rank, non_blocking=True)
        # news_input = news_input.to(local_rank, non_blocking=True)
        # hetero_graph = hetero_graph.to(local_rank, non_blocking=True)

        with amp.autocast():
            bz_loss, y_hat = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels, clicked_event, candidate_event, clicked_topic_ids, clicked_topic_ids_mask, clicked_subtopic_ids, clicked_subtopic_ids_mask, clicked_subtopic_news_ids, clicked_subtopic_news_ids_mask, clicked_event_mask, indirect_entity_neighbors, indirect_entity_neighbors_mask)
            
        # Accumulate the gradients
        scaler.scale(bz_loss).backward()
        if cnt % cfg.accumulation_steps == 0 or cnt == int(cfg.dataset.pos_count / cfg.batch_size):
            # Update the parameters
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()
                ## https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
            optimizer.zero_grad(set_to_none=True)

        sum_loss += bz_loss.data.float()
        sum_auc += area_under_curve(labels, y_hat)
        # ---------------------------------------- Training Log
        if cnt % cfg.log_steps == 0:
            if local_rank == 0:
                wandb.log({"train_loss": sum_loss.item() / cfg.log_steps, "train_auc": sum_auc.item() / cfg.log_steps})
            print('[{}] Ed: {}, average_loss: {:.5f}, average_acc: {:.5f}'.format(
                local_rank, cnt * cfg.batch_size, sum_loss.item() / cfg.log_steps, sum_auc.item() / cfg.log_steps))
            sum_loss.zero_()
            sum_auc.zero_()
        
        if cnt > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)) and  cnt % cfg.val_steps == 0:
            res = val(model, local_rank, cfg)
            model.train()

            if local_rank == 0:
                pretty_print(res)
                wandb.log(res)

            early_stop, get_better = early_stopping(res['auc'])
            if early_stop:
                print("Early Stop.")
                break
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc']}")
                    wandb.run.summary.update({"best_auc": res["auc"],"best_mrr":res['mrr'], 
                                         "best_ndcg5":res['ndcg5'], "best_ndcg10":res['ndcg10']})


def val(model, local_rank, cfg):
    model.eval()
    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank)
    tasks = []
    with torch.no_grad():
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, clicked_event, candidate_event, clicked_topic_ids, clicked_topic_ids_mask, clicked_subtopic_ids, clicked_subtopic_ids_mask, clicked_subtopic_news_list, clicked_subtopic_news_ids_mask, clicked_event_mask, indirect_entity_neighbors, indirect_entity_neighbors_mask) \
                in enumerate(tqdm(dataloader,
                                  total=int(cfg.dataset.val_len / cfg.gpu_num ),
                                  # total=int(cfg.dataset.val_len / 1 ),
                                  desc=f"[{local_rank}] Validating")):
            # print(f"clicked_entity.shape: {clicked_entity.shape}")
            # print(f"candidate_input.shape: {candidate_input.shape}")
            # print(f"candidate_entity.shape: {candidate_entity.shape}")
            # print(f"clicked_event.shape: {clicked_event.shape}")
            # print(f"clicked_topic_ids.shape: {clicked_topic_ids.shape}")
            # print(f"clicked_topic_ids_mask.shape: {clicked_topic_ids_mask.shape}")
            # print(f"clicked_subtopic_ids.shape: {clicked_subtopic_ids.shape}")
            # print(f"clicked_subtopic_ids_mask.shape: {clicked_subtopic_ids_mask.shape}")
            # print(f"clicked_subtopic_news_ids.shape: {clicked_subtopic_news_ids.shape}")
            # print(f"clicked_subtopic_news_ids_mask.shape: {clicked_subtopic_news_ids_mask.shape}")
            # print(f"subgraph.x.shape: {subgraph.x.shape}")
            # print(f"candidate_input.shape: {candidate_input.shape}")
            # print(f"clicked_event.shape: {clicked_event.shape}")
            # print(f"in main, clicked_subtopic_news_ids: {clicked_subtopic_news_ids}")
            # print(f"[val] clicked_subtopic_news_list.shape: {clicked_subtopic_news_list.shape}")
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)
            # clicked_abs_entity = clicked_abs_entity.to(local_rank, non_blocking=True)
            # clicked_subcategory = clicked_subcategory.to(local_rank, non_blocking=True)


            clicked_event = clicked_event.to(local_rank, non_blocking=True)
            clicked_event_mask = clicked_event_mask.to(local_rank, non_blocking=True)
            candidate_event = candidate_event.to(local_rank, non_blocking=True)

            # hie_emb = model.module.hieRec_encoder(clicked_topic_ids, clicked_topic_ids_mask, clicked_subtopic_ids, clicked_subtopic_ids_mask, clicked_subtopic_news_ids, clicked_subtopic_news_ids_mask)
            # print(f"hie_emb.shape: {hie_emb.shape}")
            # clicked_event_mask = clicked_event_mask.to(local_rank, non_blocking=True)
            # if len(clicked_key_entity) != 0:
            #     clicked_key_entity = clicked_key_entity.to(local_rank, non_blocking=True)
            #     clicked_key_entity_mask = clicked_key_entity_mask.to(local_rank, non_blocking=True)
            # if len(candidate_key_entity) != 0:
            #     candidate_key_entity = candidate_key_entity.to(local_rank, non_blocking=True)
            #     candidate_key_entity_mask = candidate_key_entity_mask.to(local_rank, non_blocking=True)
            # if len(abs_candidate_entity) != 0:
            #     candidate_key_entity = candidate_key_entity.to(local_rank, non_blocking=True)
            # if len(abs_candidate_entity) != 0:
            #     abs_candidate_entity = abs_candidate_entity.to(local_rank, non_blocking=True)
            #     abs_entity_mask = abs_entity_mask.to(local_rank, non_blocking=True)
            # if len(subcategories) != 0:
            #     subcategories = subcategories.to(local_rank, non_blocking=True)

            clicked_topic_ids = torch.tensor(clicked_topic_ids).unsqueeze(0).to(local_rank, non_blocking=True)
            clicked_topic_ids_mask = torch.tensor(clicked_topic_ids_mask).unsqueeze(0).to(local_rank, non_blocking=True)
            # clicked_topic_news_ids = clicked_topic_news_ids.to(local_rank, non_blocking=True)
            # clicked_topic_news_ids_mask = clicked_topic_news_ids_mask.to(local_rank, non_blocking=True)
            clicked_subtopic_ids = torch.tensor(clicked_subtopic_ids).unsqueeze(0).to(local_rank, non_blocking=True)
            clicked_subtopic_ids_mask = torch.tensor(clicked_subtopic_ids_mask).unsqueeze(0).to(local_rank, non_blocking=True)
            # shape = [len(_sublist) for _sublist in sublist for sublist in clicked_subtopic_news_ids]
            # shape = []
            # for sublist in clicked_subtopic_news_ids:
            #     for lst in sublist:
            #         print(f"{len(clicked_subtopic_news_ids)} * {len(sublist)} * {len(lst)}")
            #     print()
            # print(f"shape: {shape}")
            # np_arr = np.array(clicked_subtopic_news_ids)
            # print(f"np_arr.shape: {np_arr.shape}")
            clicked_subtopic_news_list = clicked_subtopic_news_list.to(local_rank, non_blocking=True)
            clicked_subtopic_news_ids_mask = torch.tensor(clicked_subtopic_news_ids_mask).unsqueeze(0).to(local_rank, non_blocking=True)
            # candidate_second_order_entity_neighbor = candidate_second_order_entity_neighbor.to(local_rank, non_blocking=True)
            # second_order_entity_mask = second_order_entity_mask.to(local_rank, non_blocking=True)
            indirect_entity_neighbors = indirect_entity_neighbors.to(local_rank, non_blocking=True)
            indirect_entity_neighbors_mask = indirect_entity_neighbors_mask.to(local_rank, non_blocking=True)
            # print(f"clicked_subtopic_news_ids: {clicked_subtopic_news_ids}")
            # clicked_subtopic_news_lists = torch.tensor(np.array(clicked_subtopic_news_list)).to(local_rank, non_blocking=True)
            # hetero_graph = hetero_graph.to(local_rank, non_blocking=True)
            # user_id = user_id.to(local_rank, non_blocking=True)

            # print(f"in main.py, clicked_event.shape = {clicked_event}")
            # print(f"clicked_subtopic_news_ids.shape: {clicked_subtopic_news_ids.shape}")
            scores = model.module.validation_process(subgraph, mappings, clicked_entity, candidate_emb, candidate_entity,
                                                     entity_mask,
                                                     clicked_event, candidate_event,
                                                     clicked_topic_ids, clicked_topic_ids_mask,
                                                     clicked_subtopic_ids, clicked_subtopic_ids_mask, clicked_subtopic_news_list, clicked_subtopic_news_ids_mask, clicked_event_mask,
                                                     indirect_entity_neighbors, indirect_entity_neighbors_mask
                                                     )
            # clicked_key_entity, clicked_key_entity_mask, candidate_key_entity, candidate_key_entity_mask,

            tasks.append((labels.tolist(), scores))

    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier
    torch.distributed.barrier()

    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    # reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), 1)
    # reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), 1)
    # reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), 1)
    # reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), 1)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }

    return res


def test(model, local_rank, cfg):
    model.eval()
    dataloader = load_data(cfg, mode='test', model=model, local_rank=local_rank)
    tasks = []
    with torch.no_grad():
        for cnt, (
        subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels,
        clicked_event, candidate_event,
        clicked_event_mask, clicked_key_entity, clicked_key_entity_mask, candidate_key_entity, candidate_key_entity_mask) \
                in enumerate(tqdm(dataloader,
                                  total=int(cfg.dataset.test_len / cfg.gpu_num),
                                  # total=int(cfg.dataset.val_len / 1 ),
                                  desc=f"[{local_rank}] Testing")):
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)
            # clicked_abs_entity = clicked_abs_entity.to(local_rank, non_blocking=True)
            # clicked_subcategory = clicked_subcategory.to(local_rank, non_blocking=True)

            clicked_event = clicked_event.to(local_rank, non_blocking=True)
            candidate_event = candidate_event.to(local_rank, non_blocking=True)
            clicked_event_mask = clicked_event_mask.to(local_rank, non_blocking=True)
            if len(clicked_key_entity) != 0:
                clicked_key_entity = clicked_key_entity.to(local_rank, non_blocking=True)
                clicked_key_entity_mask = clicked_key_entity_mask.to(local_rank, non_blocking=True)
            if len(candidate_key_entity) != 0:
                candidate_key_entity = candidate_key_entity.to(local_rank, non_blocking=True)
                candidate_key_entity_mask = candidate_key_entity_mask.to(local_rank, non_blocking=True)
            # if len(abs_candidate_entity) != 0:
            #     abs_candidate_entity = abs_candidate_entity.to(local_rank, non_blocking=True)
            #     abs_entity_mask = abs_entity_mask.to(local_rank, non_blocking=True)
            # if len(subcategories) != 0:
            #     subcategories = subcategories.to(local_rank, non_blocking=True)

            # print(f"in main.py, clicked_event.shape = {clicked_event}")
            scores = model.module.validation_process(subgraph, mappings, clicked_entity, candidate_emb,
                                                     candidate_entity, entity_mask,
                                                     clicked_event, candidate_event, clicked_event_mask,
                                                     clicked_key_entity, clicked_key_entity_mask, candidate_key_entity, candidate_key_entity_mask)

            tasks.append((labels.tolist(), scores))

    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier
    torch.distributed.barrier()

    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    # reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), 1)
    # reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), 1)
    # reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), 1)
    # reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), 1)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }

    if local_rank == 0:
        pretty_print(res)
        wandb.log(res)
        wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                                  "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})

    return res


def main_worker(local_rank, cfg):
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=cfg.gpu_num,
                            rank=local_rank)

    # -----------------------------------------Dataset & Model Load
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)
    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank)
    model = load_model(cfg).to(local_rank)
    # print(f"model.parameters: {model.parameters}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    lr_lambda = lambda step: 1.0 if step > num_warmup_steps else step / num_warmup_steps
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # ------------------------------------------Load Checkpoint & optimizer
    if cfg.load_checkpoint:
        file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}.pth")
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # After Distributed
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer.zero_grad(set_to_none=True)
    scaler = amp.GradScaler()

    # ------------------------------------------Main Start
    early_stopping = EarlyStopping(cfg.early_stop_patience)

    if local_rank == 0:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                   project=cfg.logger.exp_name, name=cfg.logger.run_name)
        print(model)

    # for _ in tqdm(range(1, cfg.num_epochs + 1), desc="Epoch"): NO THIS LINE

    # TODO train
    train(model, optimizer, scaler, scheduler, train_dataloader, local_rank, cfg, early_stopping)
    # res = test(model, local_rank, cfg)
    # print(f"res: {res}")

    if local_rank == 0:
        wandb.finish()



@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
# @hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="large")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    cfg.gpu_num = torch.cuda.device_count()
    prepare_preprocessed_data(cfg)
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))


if __name__ == "__main__":
    main()

