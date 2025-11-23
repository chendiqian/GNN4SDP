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
from utils.experiment import setup_wandb, sync_timer

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="ppgn")
def main(args: DictConfig):
    setup_wandb(args)

    test_set = LPDataset(args.train.datapath, 'train', transform=None)

    if args.train.debug:
        test_set = test_set[-2:]

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'
    wandb.log({'device': device})
    logger.info(f"Using device: {device}")

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

    repeats = 3

    pbar = tqdm(test_set)
    for data in pbar:
        name= data.name
        A, C, b = recover_sdp_from_data(data)
        n = C.shape[0]
        m = A.shape[-1]
        times = []
        for r in range(repeats):
            *_, sol = solve_sdp_scs(C, A, b, regularization=1.e-5, verbose=False, gpu=use_gpu)
            times.append(sol['info']['solve_time'])
            logger.info(f"repeat {r}: solver time: {sol['info']['solve_time']}")
        times = np.array(times) / 1000.
        wandb.log({f'{name}_solver_time_mean': np.mean(times)})
        wandb.log({f'{name}_solver_time_std': np.std(times)})

        batch = collate_fn_lp_base([data]).to(device)

        warm_times = []
        gnn_times = []
        for model_dict in model_dicts:
            state_dict = torch.load(os.path.join(args.train.modelpath, model_dict), map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                t1 = sync_timer()
                pred_primal, pred_slack, pred_dual = model(batch)
                t2 = sync_timer()

            logger.info(f"GNN time: {t2 - t1}")
            gnn_times.append(t2 - t1)
            x = map_vec(pred_primal.detach().cpu().numpy().reshape(n, n, 1)).squeeze()
            s = np.hstack([np.zeros(m), x])
            y = np.hstack([pred_dual.detach().cpu().numpy(),
                           map_vec(pred_slack.detach().cpu().numpy().reshape(n, n, 1)).squeeze()])
            *_, sol = solve_sdp_scs(C, A, b, regularization=1.e-5, verbose=False, gpu=use_gpu, warm_start=True, x=x, y=y, s=s)
            logger.info(f"warm start time: {sol['info']['solve_time']}")
            warm_times.append(sol['info']['solve_time'])

        warm_times = np.array(warm_times) / 1000
        gnn_times = np.array(gnn_times)

        wandb.log({f'{name}_gnn_mean': np.mean(gnn_times)})
        wandb.log({f'{name}_gnn_std': np.std(gnn_times)})

        wandb.log({f'{name}_warm_mean': np.mean(warm_times)})
        wandb.log({f'{name}_warm_std': np.std(warm_times)})


if __name__ == '__main__':
    main()
