import hydra
import numpy as np
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import LPDataset
from utils.evaluation import recover_sdp_from_data, solve_sdp_cvxpy
from utils.experiment import setup_wandb, sync_timer


@hydra.main(version_base=None, config_path='./config', config_name="mpnn")
def main(args: DictConfig):
    setup_wandb(args)

    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        test_set = test_set[:20]

    A, C, b = recover_sdp_from_data(test_set[0])
    for _ in range(5):
        # warm start
        _ = solve_sdp_cvxpy(C, A, b, 1.e-3)

    times = []
    for data in tqdm(test_set):
        A, C, b = recover_sdp_from_data(data)
        t1 = sync_timer()
        _ = solve_sdp_cvxpy(C, A, b, 1.e-3)
        t2 = sync_timer()
        times.append(t2 - t1)

    wandb.log({
        'time_mean': np.mean(times),
        'time_std': np.std(times),
    })


if __name__ == '__main__':
    main()
