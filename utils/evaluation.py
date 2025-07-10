from typing import List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor, spmm


# def recover_qp_from_data(data, dtype=np.float32):
#     data = data.to('cpu')
#     c = data.q.numpy().astype(dtype)
#     b = data.b.numpy().astype(dtype)
#     A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
#                      col=data['cons', 'to', 'vals'].edge_index[1],
#                      value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
#                      sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
#     P = SparseTensor(row=data['vals', 'to', 'vals'].edge_index[0],
#                      col=data['vals', 'to', 'vals'].edge_index[1],
#                      value=data['vals', 'to', 'vals'].edge_attr.squeeze(),
#                      sparse_sizes=(data['vals'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
#     lb = np.zeros(A.shape[1]).astype(dtype)
#     ub = None
#     return P, A, c, b, lb, ub


def normalize_cons(A, b):
    if A is None or b is None:
        return A, b
    Ab = np.concatenate([A, b[:, None]], axis=1)
    max_logit = np.abs(Ab).max(axis=1)
    max_logit[max_logit == 0] = 1.
    Ab = Ab / max_logit[:, None]
    A = Ab[:, :-1]
    b = Ab[:, -1]
    return A, b
