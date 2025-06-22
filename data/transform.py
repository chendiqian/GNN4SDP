import warnings

import torch
from torch_geometric.data import HeteroData


class SelectEigvec:
    def __init__(self, k):
        assert k >= 0
        self.k = k

    def __call__(self, data: HeteroData) -> HeteroData:
        # warning: nnodes is the square of val nodes
        n = int(data['vals'].num_nodes ** 0.5)
        device = data['vals'].x.device
        if not hasattr(data, 'c_eigval'):
            warnings.warn("Eigval not detected")
            data.c_eigvec = torch.ones(n, 1, device=device).float()
            data.c_eigval = torch.ones(1, device=device).float()
        elif self.k == 0:
            # not using the eigvals
            data.c_eigvec = torch.ones(n, 1, device=device).float()
            data.c_eigval = torch.ones(1, device=device).float()
        elif self.k <= data.c_eigval.shape[-1]:
            # slice
            data.c_eigvec = data.c_eigvec[:, -self.k:]
            data.c_eigval = data.c_eigval[-self.k:]
        else:
            data.c_eigvec = torch.hstack([data.c_eigvec,
                                          torch.zeros(n, self.k - data.c_eigvec.shape[1], device=device).float()])
            data.c_eigval = torch.hstack([data.c_eigval,
                                          torch.zeros(self.k - data.c_eigvec.shape[1], device=device).float()])

        data.c_eigval = data.c_eigval[None, :].repeat(n, 1)
        return data
