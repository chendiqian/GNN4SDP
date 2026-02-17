import torch
from torch_geometric.nn import MLP


def contractions_2_to_1(inputs):
    B, N, _, F = inputs.shape
    diag_indices = torch.arange(N, device=inputs.device)

    diag_part = inputs[:, diag_indices, diag_indices, :]  # B N F
    sum_of_rows = torch.mean(inputs, dim=2)  # B N F
    return torch.stack([diag_part, sum_of_rows], dim=0)


class Layer2to1(torch.nn.Module):
    def __init__(self, input_depth, output_depth, mlp_layers, act):
        super().__init__()

        basis_dimension = 2

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(
            torch.empty(basis_dimension, input_depth, input_depth), requires_grad=True)
        torch.nn.init.xavier_normal_(self.coeffs)

        # bias
        self.all_bias = torch.nn.Parameter(torch.zeros(1, 1, input_depth))

        self.mlp = MLP([input_depth] * mlp_layers + [output_depth], act=act, norm=None)

    def forward(self, inputs, *args, **kwargs):
        """
        :param inputs: N x m x m x D tensor
        :return: output: N x m x S tensor
        """
        ops_out = contractions_2_to_1(inputs)
        output = torch.einsum('dfh,dbnf->bnh', self.coeffs, ops_out)

        # bias
        output = output + self.all_bias

        return self.mlp(torch.relu(output))
