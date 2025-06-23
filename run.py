import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from utils.experiment import save_run_config, setup_wandb, count_parameters
from models.hetero_gnn import GNN
from trainers.supervised_trainer import PlainGNNTrainer
from trainers.training_loops import supervised_train_eval_loops
from data.transform import SelectEigvec


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    transform = SelectEigvec(args.gnn.num_eig)
    train_set = LPDataset(args.train.datapath, 'train', transform=transform)
    valid_set = LPDataset(args.train.datapath, 'valid', transform=transform)
    test_set = LPDataset(args.train.datapath, 'test', transform=transform)

    if args.train.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    train_loader = DataLoader(train_set,
                              batch_size=args.train.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn_lp_base,
                              pin_memory=True)
    val_loader = DataLoader(valid_set,
                            batch_size=args.train.batchsize * 2,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.train.batchsize * 2,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base,
                             pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.train.runs):
        model = GNN(hid_dim=args.gnn.hidden,
                    num_encode_layers=args.gnn.num_encode_layers,
                    num_conv_layers=args.gnn.num_conv_layers,
                    num_pred_layers=args.gnn.num_pred_layers,
                    num_mlp_layers=args.gnn.num_mlp_layers,
                    ign_mlp_layer=args.gnn.ign_mlp_layer,
                    norm=args.gnn.norm,
                    act=args.gnn.act).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=100,
                                                         min_lr=1.e-5)

        trainer = PlainGNNTrainer()

        best_model = supervised_train_eval_loops(args.train.epoch, args.train.patience, args.train.ckpt,
                                                 run, log_folder_name,
                                                 trainer, train_loader, val_loader, model, optimizer, scheduler)
        model.load_state_dict(best_model)
        _, test_obj_gap = trainer.eval(test_loader, model)

        best_val_objgaps.append(np.mean(trainer.best_objgap))
        test_objgaps.append(np.mean(test_obj_gap))

    wandb.log({
        'num_params': count_parameters(model),
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps)
    })


if __name__ == '__main__':
    main()
