import cvxpy as cp
import numpy as np
from torch_sparse import SparseTensor


def recover_sdp_from_data(data, dtype=np.float32):
    data = data.to('cpu')
    b = data.b.numpy().astype(dtype)
    # m * n^2
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
    m = A.shape[0]
    n = int(A.shape[1] ** 0.5)
    A = A.T.reshape(n, n, m)

    C = SparseTensor(row=data['obj', 'to', 'vals'].edge_index[0],
                     col=data['obj', 'to', 'vals'].edge_index[1],
                     value=data['obj', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(1, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
    C = C.squeeze(0).reshape(n, n)

    return A, C, b


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


def solve_sdp_cvxpy(C, A, b, norm_strength=0., solver='mosek'):
    N = C.shape[0]
    M = A.shape[-1]
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((N, N), PSD=True)

    # The operator >> denotes matrix inequality.
    # constraints = [X >> 0]
    constraints = [cp.trace(A[..., i] @ X) == b[i] for i in range(M)]
    objective = cp.trace(C @ X)
    # wrt the min norm
    if norm_strength > 0:
        objective += cp.sum_squares(X) * norm_strength
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(verbose=False, solver=getattr(cp, solver.upper()))

    # Print result.
    sol = prob.value
    X = X.value

    return sol, X, prob.status, prob.solver_stats.solve_time
