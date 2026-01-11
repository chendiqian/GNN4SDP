import os

from loguru import logger
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from models import get_model
from trainer import PlainGNNTrainer
from utils.experiment import save_run_config, setup_wandb

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="ppgn")
def main(args: DictConfig):
    setup_wandb(args)

    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        test_set = test_set[:20]

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base,
                             pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_losses = []
    test_objgaps = []
    psd_obj_gaps = []
    test_vios = []
    psd_vios = []

    model_dicts = os.listdir(args.train.modelpath)
    model_dicts = [m for m in model_dicts if m.startswith('best') and m.endswith('.pt')]

    for run, model_dict in enumerate(model_dicts):
        model = get_model(args.gnn).to(device)
        state_dict = torch.load(os.path.join(args.train.modelpath, model_dict), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        trainer = PlainGNNTrainer(args.train.accum)

        test_loss, test_obj_gap, psd_obj_gap, vio, psd_vio = trainer.eval(test_loader, model, device, True)

        test_losses.append(test_loss.item())
        test_objgaps.append(test_obj_gap.item())
        psd_obj_gaps.append(psd_obj_gap.item())
        test_vios.append(vio.item())
        psd_vios.append(psd_vio.item())

    stats = {
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
        'test_psd_obj_gap_mean': np.mean(psd_obj_gaps),
        'test_psd_obj_gap_std': np.std(psd_obj_gaps),
        'test_vio_mean': np.mean(test_vios),
        'test_vio_std': np.std(test_vios),
        'test_psd_vio_mean': np.mean(psd_vios),
        'test_psd_vio_std': np.std(psd_vios),
    }

    logger.info(', '.join([k + f':{v:.5f}' for k, v in stats.items()]))
    wandb.log(stats)


if __name__ == '__main__':
    main()
