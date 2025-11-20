# MPNN for general SDPs

## Environment setup

```angular2html
conda create -y -n gnnsdp python=3.11
conda activate gnnsdp

pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

pip install wandb hydra-core loguru
pip install "cvxpy[MOSEK]"
# you may need a license for the MOSEK solver
```

## Replicate results in Table 1
`python run.py --config-name ppgn train.datapath=DATAPATH + your args`

Replace `ppgn` with `edge_gt/two_wl/ign/mpnn` for other baselines. 

If Edge GT runs OOM, use batch gradient accumulation by setting a smaller batch size and `train.accum > 1`.

## replicate runtime

## test on a pretrained model
`python test.py --config-name ppgn_test train.datapath=DATAPATH train.modelpath=MODELPATH`

## Ablation on encoder
Set `gnn.encode_type=multiset` or `gnn.encode_type=cat`