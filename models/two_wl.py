import torch
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch

from models.hetero_base_nn import BaseModel


class GINEConv(torch.nn.Module):
    def __init__(self, hid_dim, num_mlp_layers, act):
        super().__init__()

        self.lin_src = MLP([hid_dim] * 2, act=act, norm=None, plain_last=False)
        self.lin_dst = MLP([hid_dim * 2, hid_dim], act=act, norm=None, plain_last=False)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), act=act, norm=None, plain_last=False)
        self.eps = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, inputs):
        # B x N x N x F
        x = self.lin_src(inputs)
        n = x.shape[1]
        aggr_x = x.mean(1, keepdim=True).repeat(1, n, 1, 1)  # B x 1 x N x F
        aggr_tuple = torch.cat([aggr_x, aggr_x.transpose(1, 2)], dim=-1)  # the 2WL tuple
        msg = self.lin_dst(aggr_tuple)
        msg = msg + msg.transpose(1, 2)
        x_dst = (1 + self.eps) * inputs + msg
        return self.mlp(x_dst)


class TwoWL(BaseModel):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 block_mlp_layers,
                 norm,
                 act):
        super().__init__(hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         num_mlp_layers,
                         norm,
                         act)

        self.two_wls = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.two_wls.append(GINEConv(hid_dim, block_mlp_layers, act))

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

            x_x_dense = self.two_wls[i](x_x_dense)
            x_x_dense = x_x_dense[real_x_x_mask]

            x_dict['vals'] = x_x_dense  # sum(nnodes ** 2) x F
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        return self.predictor(x_dict['vals']).squeeze()
