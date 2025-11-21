import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from models import get_model
from utils.experiment import setup_wandb, sync_timer

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

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        warnings.warn("No cuda available")
        device = 'cpu'

    model = get_model(args.gnn).to(device)
    with torch.no_grad():
        for data in test_loader:
            # warm sta GPU
            data = data.to(device)
            sync_timer()
            _ = model(data)
            sync_timer()

    times = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            t1 = sync_timer()
            _ = model(data)
            t2 = sync_timer()
            times.append(t2 - t1)

    wandb.log({
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'time_string': f'{np.mean(times):.3f}±{np.std(times):.3f}'
    })


if __name__ == '__main__':
    main()
