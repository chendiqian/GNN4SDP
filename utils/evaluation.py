from typing import List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor, spmm


def gurobi_solve_lp(A, b, c, lb=0., ub=float('inf')):
    import gurobipy as gp
    from gurobipy import GRB

    m, n = A.shape
    model = gp.Model("lp")
    model.Params.LogToConsole = 0
    variables = model.addMVar(n, lb=lb, ub=ub)

    # Objective: 0.5 x^T P x + q^T x
    model.setObjective(c @ variables, GRB.MINIMIZE)

    # Add inequality constraints
    constrs = model.addConstr(A @ variables <= b)

    # Solve
    model.optimize()

    # Duals
    if model.status == GRB.OPTIMAL:
        duals = constrs.getAttr("Pi")
        solution = variables.X
    else:
        duals = solution = None
    return solution, duals


def gurobi_solve_qp(Q, c, Aub, bub, Aeq=None, beq=None, lb=0., ub=float('inf')):
    import gurobipy as gp
    from gurobipy import GRB

    _, n = Aub.shape
    model = gp.Model("qp")
    model.Params.LogToConsole = 0
    variables = model.addMVar(n, lb=lb, ub=ub)

    # Objective: 0.5 x^T P x + q^T x
    model.setObjective(0.5 * variables @ Q @ variables + c @ variables)

    # Add inequality constraints
    constrs = model.addConstr(Aub @ variables <= bub)
    if Aeq is not None:
        constrs2 = model.addConstr(Aeq @ variables == beq)

    # Solve
    model.optimize()

    # Duals
    if model.status == GRB.OPTIMAL:
        duals = constrs.getAttr("Pi")
        if Aeq is not None:
            duals = np.hstack([duals, constrs2.getAttr("Pi")])
        solution = variables.X
    else:
        duals = solution = None
    return solution, duals


# def recover_lp_from_data(data, dtype=np.float32):
#     data = data.to('cpu')
#     c = data.q.numpy().astype(dtype)
#     b = data.b.numpy().astype(dtype)
#     A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
#                      col=data['cons', 'to', 'vals'].edge_index[1],
#                      value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
#                      sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
#     # todo: might vary
#     lb = np.zeros(A.shape[1]).astype(dtype)
#     ub = None
#     return A, c, b, lb, ub
#
#
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
#     # todo: might vary
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


# def calc_violation(pred, data):
#     assert pred.dim() <= 2
#     if pred.dim() == 2:
#         assert pred.shape[1] == 1
#     if pred.dim() == 1:
#         pred = pred[:, None]
#     Ax_minus_b = spmm(data['cons', 'to', 'vals'].edge_index,
#                       data['cons', 'to', 'vals'].edge_attr.squeeze(),
#                       data['cons'].num_nodes, data['vals'].num_nodes, pred).squeeze() - data.b
#     violation = scatter(torch.relu(Ax_minus_b), data['cons'].batch, dim=0, reduce='mean')  # (batchsize,)
#     return violation
