import copy
import os
from collections import defaultdict

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from models import get_model
from trainer import PlainGNNTrainer
from utils.experiment import save_run_config, setup_wandb, count_parameters

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="mpnn")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    train_set = LPDataset(args.train.datapath, 'train', transform=None)

    if args.train.debug:
        train_set = train_set[-2:]

    train_loader = DataLoader(train_set,
                              batch_size=args.train.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn_lp_base,
                              pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_dict = defaultdict(list)

    for run in range(args.train.runs):
        model = get_model(args.gnn).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=int(args.train.patience * 0.6),
                                                         min_lr=1.e-5)

        trainer = PlainGNNTrainer(args.train.accum)

        pbar = tqdm(range(args.train.epoch))
        for epoch in pbar:
            _ = trainer.train(train_loader, model, optimizer, device).item()
            train_loss, train_obj_gap, _, _, _ = trainer.eval(train_loader, model, device, False)
            train_loss = train_loss.item()
            train_obj_gap = train_obj_gap.item()

            if scheduler is not None:
                scheduler.step(train_obj_gap)

            if trainer.best_objgap > train_obj_gap:
                trainer.patience = 0
                trainer.best_objgap = train_obj_gap
                best_model = copy.deepcopy(model.state_dict())
                if args.train.ckpt:
                    torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
            else:
                trainer.patience += 1

            if trainer.patience > args.train.patience:
                break

            stats_dict = {'train_loss': train_loss,
                          'train_obj_gap': train_obj_gap,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)

        model.load_state_dict(best_model)

        for graph in train_set:
            test_loader = DataLoader([graph],
                                     batch_size=1,
                                     shuffle=False,
                                     collate_fn=collate_fn_lp_base)
            name = graph.name
            test_loss, test_obj_gap, psd_obj_gap, _, _ = trainer.eval(test_loader, model, device, True)
            result_dict[f'{name}_loss'].append(test_loss.item())
            result_dict[f'{name}_objgap'].append(test_obj_gap.item())
            result_dict[f'{name}_psdobjgap'].append(psd_obj_gap.item())

    result_stats = {'num_params': count_parameters(model)}
    for k, v in result_dict.items():
        result_stats[k + '_mean'] = np.mean(v).item()
        result_stats[k + '_std'] = np.std(v).item()

    wandb.log(result_stats)


if __name__ == '__main__':
    main()
