import torch
import torch.nn as nn
from torch_geometric.nn.resolver import activation_resolver

from models.hetero_higher_order import HigherOrder


# https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch/blob/master/layers/equivariant_linear_pytorch.py
@torch.compile
def contractions_2_to_2(inputs):
    B, N, _, F = inputs.shape
    diag_indices = torch.arange(N, device=inputs.device)

    diag_part = inputs[:, diag_indices, diag_indices, :].transpose(1, 2)  # B F N
    sum_diag_part = torch.mean(diag_part, dim=2, keepdim=True)  # B F 1
    sum_of_rows = torch.mean(inputs, dim=2)  # B N F
    sum_all = torch.mean(sum_of_rows, dim=1)  # B F

    # op1 - (1234) - extract diag
    op1 = torch.diag_embed(diag_part, dim1=1, dim2=2)

    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = torch.diag_embed(sum_diag_part.expand(B, F, N), dim1=1, dim2=2)

    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = torch.diag_embed(sum_of_rows.transpose(1, 2), dim1=1, dim2=2)

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op4 = torch.diag_embed(sum_all.unsqueeze(dim=2).expand(B, F, N), dim1=1, dim2=2)

    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op5 = sum_of_rows.unsqueeze(dim=1).expand(B, N, N, F)
    op5 = (op5 + op5.transpose(1, 2)) / 2

    # op10 - (1234) + (14)(23) - identity
    op6 = inputs  # N x D x m x m

    # op12 - (1234) + (234)(1) - place ii element in row i
    op7 = diag_part.transpose(1, 2).unsqueeze(dim=1).expand(B, N, N, F)
    op7 = (op7 + op7.transpose(1, 2)) / 2

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op8 = sum_diag_part.transpose(1, 2).unsqueeze(1).expand(B, N, N, F)

    # op15 - sum of all ops - place sum of all entries in all entries
    op9 = sum_all[:, None, None, :].expand(B, N, N, F)

    return torch.stack([op1, op2, op3, op4, op5, op6, op7, op8, op9], dim=0)


class Layer2to2(nn.Module):
    def __init__(self, input_depth, output_depth, act):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.act = activation_resolver(act)

        self.basis_dimension = 9

        # initialization values for variables
        self.coeffs = nn.Parameter(
            torch.empty(self.basis_dimension, self.input_depth, self.output_depth), requires_grad=True)
        nn.init.xavier_normal_(self.coeffs)

        # bias
        self.diag_bias = torch.nn.Parameter(torch.zeros(1, 1, 1, self.output_depth))
        self.all_bias = torch.nn.Parameter(torch.zeros(1, 1, 1, self.output_depth))

    def forward(self, inputs, *args, **kwargs):
        """
        :param inputs: N x m x m x D tensor
        :return: output: N x m x m x S tensor
        """
        ops_out = contractions_2_to_2(inputs)
        output = torch.einsum('dfh,dbnmf->bnmh', self.coeffs, ops_out)

        # bias
        mat_diag_bias = torch.eye(inputs.size(1), device=output.device)[None, :, :, None] * self.diag_bias
        output = output + self.all_bias + mat_diag_bias

        return self.act(output)


class IGN(HigherOrder):
    def __init__(self,
                 no_mp,
                 no_wl,
                 no_dual,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 gnn_mlp_layers,
                 num_pred_layers,
                 norm,
                 act):
        super().__init__(no_mp,
                         no_wl,
                         no_dual,
                         hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         gnn_mlp_layers,
                         num_pred_layers,
                         norm,
                         act)

        if not no_wl:
            self.init_higher_order_layers(num_conv_layers, hid_dim, act)

    def init_higher_order_layers(self, num_conv_layers, hid_dim, act):
        self.higher_orders = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.higher_orders.append(Layer2to2(hid_dim, hid_dim, act))
