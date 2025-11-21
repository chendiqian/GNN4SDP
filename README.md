# MPNN for general SDPs

## Environment setup

```angular2html
conda create -y -n gnnsdp python=3.11
conda activate gnnsdp

pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

pip install wandb hydra-core loguru
```

### (Optional) Solvers

For CVXPY with MOSEK solver, `pip install "cvxpy[MOSEK]"`. Note that you may need a license for the MOSEK solver.

For SCS solver with CPU usage only, `pip install scs`. For GPU acceleration, see the official documentation [SCS install](https://www.cvxgrp.org/scs/install/python.html#python-install), and a building example on [Colab](https://colab.research.google.com/drive/1POCgDNFg8fycHMI9T9N6V3iHFhXRthjn?usp=sharing). 

## Replicate results in Table 1
`python run.py --config-name ppgn train.datapath=DATAPATH + your args`

Replace `ppgn` with `edge_gt/two_wl/ign/mpnn` for other baselines. 

If Edge GT runs OOM, use batch gradient accumulation by setting a smaller batch size and `train.accum > 1`.

## Replicate runtime

For CVXPY with MOSEK solver or SCS solver (built-in in CVXPY), run  
`python timing_solver.py --config-name solver solver=mosek train.datapath="/PATH/TO/DATASET"`

For timing the neural architectures, run  
`python timing.py --config-name ppgn train.datapath="/PATH/TO/DATASET"` plus some extra `gnn` parameters. The config `ppgn` can be replaced by `edge_gt`, `two_wl` etc. in `./config` folder. 

For warm starting the SCS solver, run  
`python warm_start_scs.py --config-name ppgn +train.modelpath="/PATH/TO/MODEL_FOLDER" train.datapath="/PATH/TO/DATASET" gnn.no_dual=false` plus some extra `gnn` parameters. Note that a pretrained model directory `"/PATH/TO/MODEL_FOLDER"` is necessary. 
