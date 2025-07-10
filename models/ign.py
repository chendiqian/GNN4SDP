import torch
import torch.nn as nn
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch

from models.hetero_base_nn import BaseModel


# https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch/blob/master/layers/equivariant_linear_pytorch.py
def contractions_2_to_2(inputs, dim):  # N x m x m x D
    diag_part = torch.diagonal(inputs, dim1=1, dim2=2)  # N x D x m
    sum_diag_part = torch.mean(diag_part, dim=2, keepdim=True)  # N x D x 1
    sum_of_rows = torch.mean(inputs, dim=2)  # N x m x D
    sum_all = torch.mean(sum_of_rows, dim=1)  # N x D

    # op1 - (1234) - extract diag
    op1 = torch.diag_embed(diag_part, dim1=1, dim2=2)  # N x m x m x D

    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim), dim1=1, dim2=2)  # N x m x m x D

    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = torch.diag_embed(sum_of_rows.transpose(1, 2), dim1=1, dim2=2)  # N x m x m x D

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op4 = torch.diag_embed(sum_all.unsqueeze(dim=2).repeat(1, 1, dim), dim1=1, dim2=2)  # N x m x m x D

    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op5 = sum_of_rows.unsqueeze(dim=1).repeat(1, dim, 1, 1)  # N x m x m x D
    op5 = (op5 + op5.transpose(1, 2)) / 2

    # op10 - (1234) + (14)(23) - identity
    op6 = inputs  # N x D x m x m

    # op12 - (1234) + (234)(1) - place ii element in row i
    op7 = diag_part.transpose(1, 2).unsqueeze(dim=1).repeat(1, dim, 1, 1)
    op7 = (op7 + op7.transpose(1, 2)) / 2

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op8 = sum_diag_part.transpose(1, 2).unsqueeze(1).repeat(1, dim, dim, 1)

    # op15 - sum of all ops - place sum of all entries in all entries
    op9 = sum_all[:, None, None, :].repeat(1, dim, dim, 1)

    return [op1, op2, op3, op4, op5, op6, op7, op8, op9]


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

    @torch.compile
    def forward(self, inputs):
        """
        :param inputs: N x m x m x D tensor
        :return: output: N x m x m x S tensor
        """
        m = inputs.size(1)  # extract dimension

        ops_out = contractions_2_to_2(inputs, m)
        ops_out = torch.stack(ops_out, dim=0)

        output = torch.einsum('dfh,dbnmf->bnmh', self.coeffs, ops_out)

        # bias
        mat_diag_bias = torch.eye(inputs.size(1), device=output.device)[None, :, :, None] * self.diag_bias
        output = output + self.all_bias + mat_diag_bias

        return self.act(output)


class IGN(BaseModel):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act):
        super().__init__(hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         num_mlp_layers,
                         norm,
                         act)

        self.igns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.igns.append(Layer2to2(hid_dim, hid_dim, act))

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        _, real_x_mask = to_dense_batch(x_dict['vals'].new_empty(batch_dict['_vals'].shape[0]),
                                        batch_dict['_vals'])  # B x Nmax x F
        real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax
        feature_dim = x_dict['vals'].shape[-1]
        device = x_dict['vals'].device

        for i, layer in enumerate(self.gcns):
            x_x_dense = torch.zeros(*real_x_x_mask.shape + (feature_dim,), device=device, dtype=torch.float)
            x_x_dense[real_x_x_mask] = x_dict['vals']

            x_x_dense = self.igns[i](x_x_dense)
            x_x_dense = x_x_dense[real_x_x_mask]

            x_dict['vals'] = x_x_dense  # sum(nnodes ** 2) x F
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        return self.predictor(x_dict['vals']).squeeze()
