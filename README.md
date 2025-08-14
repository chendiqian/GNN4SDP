# MPNN for general SDPs

## Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate gnnsdp
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb hydra-core loguru
pip install "cvxpy[MOSEK]"
# you may need a license for the MOSEK solver
```

## Replicate results in Table 1
For Max Cut, Synthetic data, Max2SAT problems, we use 1 GNN encoder layer:  
`python run.py --config-name ppgn train.datapath=DATAPATH gnn.num_gnn_layers=1`

For Set Cover, Max Independent Set and Max3SAT problem, we use 2 GNN encoder layers:  
`python run.py --config-name ppgn train.datapath=DATAPATH gnn.num_gnn_layers=2`

Replace `ppgn` with `edge_gt/two_wl/ign/mpnn` for other baselines. 

If Edge GT runs OOM, use batch gradient accumulation by setting a smaller batch size and `train.accum > 1`.

## replicate runtime

## test on a pretrained model
`python test.py --config-name ppgn_test train.datapath=DATAPATH train.modelpath=MODELPATH`

## Ablation on encoder
Set `gnn.encode_type=multiset` or `gnn.encode_type=cat`