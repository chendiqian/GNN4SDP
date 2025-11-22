import cvxpy as cp
import numpy as np
import scs
from scipy import sparse
from torch_sparse import SparseTensor


# The vec function as documented in api/cones
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]


def map_vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n), :] /= np.sqrt(2)
    idx = np.triu_indices(n)
    return S[idx[0], idx[1], :].T


# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


def solve_sdp_scs(C, A, b, verbose=False, gpu=False, warm_start=False, x=None, y=None, s=None):
    m = A.shape[-1]
    n = C.shape[0]
    nvec = (n + 1) * n // 2

    c = vec(C)
    A_eq = map_vec(A)
    # A_eq = np.stack([vec(A[..., i]) for i in range(m)], axis=0)

    A_sp = sparse.vstack(
        [
            # zero cone
            A_eq,
            # positive semidefinite cone
            -sparse.eye(nvec),
        ],
        format="csc",
    )
    b = np.hstack([b, np.zeros(nvec)])

    data = dict(A=A_sp, b=b, c=c)
    # zero cone: m equalities, spd cone: n times n
    cone = dict(z=m, s=n)
    # Setup workspace
    solver = scs.SCS(data, cone, verbose=verbose, gpu=gpu)
    sol = solver.solve(warm_start, x, y, s)
    if sol['info']['status'].startswith('solved'):  # allowed to be inaccurate
        X = mat(sol["x"])
        y = sol['y'][:m]
        dual = mat(sol['y'][m:])
    else:
        X = None
        y = None
        dual = None

    return X, y, dual, sol


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
