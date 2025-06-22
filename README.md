# MPNN for general SDPs

## Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate da4l2o
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_cluster-1.6.3%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb hydra-core pytorch-metric-learning

conda install -y -c conda-forge qpsolvers 
# for larger problems you need a license, please visit https://www.gurobi.com/ for more information
pip install gurobipy
```
