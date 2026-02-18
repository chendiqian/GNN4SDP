import copy
import os

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from loguru import logger
from omegaconf import DictConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from models import get_model
from trainer import SSLPrimalTrainer
from utils.experiment import save_run_config, setup_wandb, count_parameters

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="mpnn")
def main(args: DictConfig):
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    rank = int(os.environ['RANK'])  # Rank of the current process
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size > 1, "This running file for multi gpu usage only!!!!"

    dist.init_process_group(backend="nccl", device_id=local_rank)

    train_set = LPDataset(args.train.datapath, 'train', transform=None)
    valid_set = LPDataset(args.train.datapath, 'valid', transform=None)
    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set,
                              batch_size=args.train.batchsize // world_size,
                              collate_fn=collate_fn_lp_base,
                              pin_memory=True,
                              sampler=train_sampler)
    val_loader = DataLoader(valid_set,
                            batch_size=args.train.batchsize * 2 // world_size,
                            collate_fn=collate_fn_lp_base,
                            pin_memory=True,
                            sampler=val_sampler)
    test_loader = DataLoader(test_set,
                             batch_size=args.train.batchsize * 2 // world_size,
                             collate_fn=collate_fn_lp_base,
                             pin_memory=True,
                             sampler=test_sampler)
    if rank == 0:
        log_folder_name = save_run_config(args)
        setup_wandb(args)
        best_val_objs = []
        test_objs = []
        test_objgaps = []

    torch.cuda.set_device(local_rank)
    for run in range(args.train.runs):
        torch.cuda.empty_cache()
        dist.barrier()

        model = get_model(args.gnn).to(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank])
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=int(args.train.patience * 0.6),
                                                         min_lr=1.e-5)

        trainer = SSLPrimalTrainer()

        for epoch in range(args.train.epoch):
            train_sampler.set_epoch(epoch)
            train_loss = trainer.train(train_loader, model, optimizer, local_rank)
            val_obj, val_obj_gap = trainer.eval(val_loader, model, local_rank)

            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_obj, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_obj_gap, op=dist.ReduceOp.AVG)

            val_obj = val_obj.item()
            if scheduler is not None:
                scheduler.step(val_obj)

            if trainer.best_objgap > val_obj:
                trainer.patience = 0
                trainer.best_objgap = val_obj
                best_model = copy.deepcopy(model.state_dict())
                if args.train.ckpt and rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
            else:
                trainer.patience += 1

            if trainer.patience > args.train.patience:
                break

            if rank == 0:
                stats_dict = {'train_loss': train_loss,
                              'val_obj': val_obj,
                              'val_obj_gap': val_obj_gap,
                              'lr': scheduler.optimizer.param_groups[0]["lr"]}
                wandb.log(stats_dict)
                logger.info(', '.join([k + f':{v:.5f}' for k, v in stats_dict.items()]))

        dist.barrier()
        model.load_state_dict(best_model)
        test_obj, test_obj_gap = trainer.eval(test_loader, model, local_rank)

        dist.all_reduce(test_obj, op=dist.ReduceOp.AVG)
        dist.all_reduce(test_obj_gap, op=dist.ReduceOp.AVG)
        dist.barrier()
        test_obj_gap = test_obj_gap.item()
        test_obj = test_obj.item()

        if rank == 0:
            best_val_objs.append(trainer.best_objgap)
            test_objgaps.append(test_obj_gap)
            test_objs.append(test_obj)

    if rank == 0:
        wandb.log({
            'num_params': count_parameters(model),
            'best_val_obj_gap': np.mean(best_val_objs),
            'test_gap_mean': np.mean(test_objs),
            'test_gap_std': np.std(test_objs),
            'test_obj_gap_mean': np.mean(test_objgaps),
            'test_obj_gap_std': np.std(test_objgaps),
        })

    dist.barrier()
    # at the very end
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
