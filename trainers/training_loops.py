import copy
import os

import torch
import wandb
from tqdm import tqdm


def supervised_train_eval_loops(epochs, patience,
                                ckpt, run_id, log_folder_name,
                                trainer, train_loader, val_loader, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    best_model = copy.deepcopy(model.state_dict())
    for epoch in pbar:
        train_loss = trainer.train(train_loader, model, optimizer)
        val_loss, val_obj_gap, psd_obj_gap = trainer.eval(val_loader, model)

        if scheduler is not None:
            scheduler.step(psd_obj_gap)

        if trainer.best_objgap > psd_obj_gap:
            trainer.patience = 0
            trainer.best_objgap = psd_obj_gap
            best_model = copy.deepcopy(model.state_dict())
            if ckpt:
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > patience:
            break

        stats_dict = {'train_loss': train_loss,
                      'val_loss': val_loss,
                      'val_obj_gap': val_obj_gap,
                      'psd_obj_gap': psd_obj_gap,
                      'lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)
    return best_model
