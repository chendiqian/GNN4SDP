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


def call_julia_general_sdp(C, A, b):
    """
    Solves the general SDP:
        min <C, X>
        s.t. <A_i, X> = b_i  for i = 1..m
             X >= 0 (PSD)

    Args:
        C: np.ndarray of shape (n, n) representing the cost matrix
        A: np.ndarray of shape (n, n, m) representing m constraint matrices
        b: np.ndarray of shape (m,) representing constraint targets
    """

    # ---------------------------------------------------------
    # Pass Data to Julia
    # ---------------------------------------------------------
    jl.C_julia = C
    jl.A_julia = A
    jl.b_julia = b

    jl.n = C.shape[0]
    jl.m = b.shape[0]

    # ---------------------------------------------------------
    # The Julia Optimization Block (Executed via Python)
    # ---------------------------------------------------------
    jl.seval('''
        # Convert Python objects to native Julia arrays for maximum JuMP performance
        C_mat = convert(Array{Float64, 2}, C_julia)
        A_mat = convert(Array{Float64, 3}, A_julia)
        b_vec = convert(Array{Float64, 1}, b_julia)

        # Initialize the JuMP model with the SDPLR solver
        model = Model(SDPLR.Optimizer)

        # Define a symmetric PSD matrix variable X of size n x n
        @variable(model, X[1:n, 1:n], PSD)

        # Define the objective: minimize the inner product <C, X>
        # Note: dot(C, X) is much faster than tr(C * X) because it avoids allocating a new matrix
        @objective(model, Min, dot(C_mat, X))

        # Define the general affine constraints: <A_i, X> = b_i
        @constraint(model, [i=1:m], dot(A_mat[:, :, i], X) == b_vec[i])

        # Solve it!
        optimize!(model)

        # Extract the solution matrix
        X_opt_julia = value.(X)
    ''')

    # ---------------------------------------------------------
    # Back to Python
    # ---------------------------------------------------------
    # Convert the Julia Array back into a native NumPy array
    X_opt_py = np.array(jl.X_opt_julia)
    obj = (C * X_opt_py).sum()
    vio = np.abs((A * X_opt_py[:, :, None]).sum((0, 1)) - b).mean()
    return X_opt_py, obj, vio


@hydra.main(version_base=None, config_path='./config', config_name="solver")
def main(args: DictConfig):
    setup_wandb(args)

    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        test_set = test_set[:20]

    A, C, b = recover_sdp_from_data(test_set[0])
    for _ in range(5):
        # warm start
        _ = call_julia(C)

    times = []
    obj_gaps = []
    vios = []
    pbar = tqdm(test_set)
    for data in pbar:
        A, C, b = recover_sdp_from_data(data)
        n = C.shape[0]
        m = b.shape[0]
        t1 = time.time()
        X, obj, vio = call_julia_general_sdp(C, A, b)
        times.append(time.time() - t1)
        obj_gt = data.obj_solution.cpu().numpy()[0]
        obj_gaps.append(np.abs((obj - obj_gt) / (obj_gt + 1.e-5)))
        vios.append(vio)

        pbar.set_postfix({'time': times[-1], 'gap': obj_gaps[-1], 'vio': vios[-1]})

    stats = {
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'gaps_mean': np.mean(obj_gaps),
        'gaps_std': np.std(obj_gaps),
        'vios_mean': np.mean(vios),
        'vios_std': np.std(vios),
    }
    print(stats)
    wandb.log(stats)


if __name__ == '__main__':
    main()
