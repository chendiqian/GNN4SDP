import os

import hydra
import numpy as np
import torch.cuda
import wandb
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from models import get_model
from utils.evaluation import recover_sdp_from_data, solve_sdp_scs, map_vec
from utils.experiment import setup_wandb

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="ppgn")
def main(args: DictConfig):
    setup_wandb(args)

    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'
    wandb.log({'device': device})
    logger.info(f"Using device: {device}")

    if args.train.debug:
        test_set = test_set[:20]

    A, C, b = recover_sdp_from_data(test_set[0])
    # warming up GPU
    logger.info("Warming up solver")
    for _ in range(20):
        _ = solve_sdp_scs(C, A, b, verbose=False, gpu=use_gpu)

    model = get_model(args.gnn).to(device)
    model_dicts = os.listdir(args.train.modelpath)
    model_dicts = [m for m in model_dicts if m.startswith('best') and m.endswith('.pt')]

    # warming up GPU
    logger.info("Warming up GNN")
    batch = collate_fn_lp_base([test_set[0]]).to(device)
    for _ in range(20):
        _ = model(batch)

    vanilla_times = []
    warm_started_time = []
    pbar = tqdm(test_set)
    for data in pbar:
        A, C, b = recover_sdp_from_data(data)
        n = C.shape[0]
        m = A.shape[-1]
        *_, sol = solve_sdp_scs(C, A, b, False, use_gpu)
        vanilla_times.append(sol['info']['solve_time'])

        batch = collate_fn_lp_base([data]).to(device)

        warm_times = []
        for model_dict in model_dicts:
            state_dict = torch.load(os.path.join(args.train.modelpath, model_dict), map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                pred_primal, pred_slack, pred_dual = model(batch)
            x = map_vec(pred_primal.detach().cpu().numpy().reshape(n, n, 1)).squeeze()
            s = np.hstack([np.zeros(m), x])
            y = np.hstack([pred_dual.detach().cpu().numpy(),
                           map_vec(pred_slack.detach().cpu().numpy().reshape(n, n, 1)).squeeze()])
            *_, sol = solve_sdp_scs(C, A, b, False, use_gpu, True, x=x, y=y, s=s)
            warm_times.append(sol['info']['solve_time'])
        warm_started_time.append(np.mean(warm_times))

        pbar.set_postfix({'vanilla': vanilla_times[-1], "warmed up": warm_started_time[-1]})

    logger.info(f'Summary -- vanilla: {np.mean(vanilla_times):.3f}, warmed up: {np.mean(warm_started_time):.3f}')

    vanilla_times = np.array(vanilla_times, dtype=np.float32) / 1000.
    warm_started_time = np.array(warm_started_time, dtype=np.float32) / 1000.

    wandb.log({
        'vanilla_time_mean': np.mean(vanilla_times),
        'vanilla_time_std': np.std(vanilla_times),
        'warmstart_time_mean': np.mean(warm_started_time),
        'warmstart_time_std': np.std(warm_started_time),
    })


if __name__ == '__main__':
    main()
