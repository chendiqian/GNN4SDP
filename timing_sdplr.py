import pdb
import os

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

import juliapkg

# 1. Add the required Julia packages to the Python sandbox
juliapkg.add("JuMP")
juliapkg.add("SDPLR")

# 2. Resolve and install them (this might take a minute the very first time)
juliapkg.resolve()

# 3. Now it is safe to boot up the Julia runtime
from juliacall import Main as jl

jl.seval('using JuMP, SDPLR, LinearAlgebra')

import torch
from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from models import get_model
from utils.evaluation import recover_sdp_from_data
from utils.experiment import setup_wandb
import time


def call_julia(C):
    # 1. Have Julia install the required solver packages (runs once)
    # jl.seval('import Pkg; Pkg.add(["JuMP", "SDPLR"])')

    # 2. Import the Julia libraries into the Python runtime

    # ---------------------------------------------------------
    # The Julia Optimization Block (Executed via Python)
    # ---------------------------------------------------------
    # We pass the NumPy array to Julia by just assigning it
    jl.C_julia = C
    jl.n = C.shape[0]

    # We formulate and solve the problem using Julia syntax evaluated as a string
    jl.seval('''
        # Initialize the JuMP model with the SDPLR solver
        model = Model(SDPLR.Optimizer)

        # Define a symmetric PSD matrix variable X of size n x n
        @variable(model, X[1:n, 1:n], PSD)

        # Example constraints: diagonals equal 1 (Max-Cut style)
        @constraint(model, [i=1:n], X[i,i] == 1.0)

        # Define the objective: minimize <C, X>
        @objective(model, Min, tr(C_julia * X))

        # Solve it!
        optimize!(model)

        # Extract the solution matrix
        X_opt_julia = value.(X)
    ''')

    # ---------------------------------------------------------
    # Back to Python
    # ---------------------------------------------------------
    # The resulting matrix is automatically converted back to a NumPy array
    X_opt_py = np.array(jl.X_opt_julia)

    # return objective_value


def call_julia_warm(C, X):
    # 1. Have Julia install the required solver packages (runs once)
    # jl.seval('import Pkg; Pkg.add(["JuMP", "SDPLR"])')

    # 2. Import the Julia libraries into the Python runtime

    # ---------------------------------------------------------
    # The Julia Optimization Block (Executed via Python)
    # ---------------------------------------------------------
    # We pass the NumPy array to Julia by just assigning it
    jl.C_julia = C
    jl.n = C.shape[0]

    # 2. Pass it to Julia
    jl.X_warm = X

    # We formulate and solve the problem using Julia syntax evaluated as a string
    jl.seval('''
        # Initialize the JuMP model with the SDPLR solver
        model = Model(SDPLR.Optimizer)

        # Define a symmetric PSD matrix variable X of size n x n
        @variable(model, X[1:n, 1:n], PSD)

        # Example constraints: diagonals equal 1 (Max-Cut style)
        @constraint(model, [i=1:n], X[i,i] == 1.0)

        # Define the objective: minimize <C, X>
        @objective(model, Min, tr(C_julia * X))
        
        # WARM START THE SOLVER
        # We loop through the matrix and set the initial values
        for i in 1:n, j in 1:n
            set_start_value(X[i,j], X_warm[i,j])
        end

        # Solve it!
        optimize!(model)

        # Extract the solution matrix
        X_opt_julia = value.(X)
    ''')

    # ---------------------------------------------------------
    # Back to Python
    # ---------------------------------------------------------
    # The resulting matrix is automatically converted back to a NumPy array
    X_opt_py = np.array(jl.X_opt_julia)

    # return objective_value


@hydra.main(version_base=None, config_path='./config', config_name="ppgn")
def main(args: DictConfig):
    setup_wandb(args)

    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        test_set = test_set[:20]

    A, C, b = recover_sdp_from_data(test_set[0])
    for _ in range(5):
        # warm start
        call_julia(C)

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    model = get_model(args.gnn).to(device)
    model_dicts = os.listdir(args.train.modelpath)
    model_dicts = [m for m in model_dicts if m.startswith('best') and m.endswith('.pt')]

    times = []
    times_warm = []
    for data in tqdm(test_set):
        A, C, b = recover_sdp_from_data(data)
        n = C.shape[0]
        batch = collate_fn_lp_base([data]).to(device)
        t1 = time.time()
        call_julia(C)
        times.append(time.time() - t1)

        for model_dict in model_dicts:
            state_dict = torch.load(os.path.join(args.train.modelpath, model_dict), map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()

            pred_primal, pred_slack, pred_dual = model(batch)
            x = pred_primal.detach().cpu().numpy().reshape(n, n)

            t1 = time.time()
            call_julia_warm(C, x)
            times_warm.append(time.time() - t1)

    stats = {
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'time_warm_mean': np.mean(times_warm),
        'time_warm_std': np.std(times_warm),
    }
    print(stats)
    wandb.log(stats)


if __name__ == '__main__':
    main()
