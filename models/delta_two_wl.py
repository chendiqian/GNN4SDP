import torch
from torch_geometric.nn import MLP

from models.hetero_higher_order import HigherOrder
from models.util import upper_triangle_mask


class GINEConv(torch.nn.Module):
    def __init__(self, hid_dim, num_mlp_layers, act):
        super().__init__()

        self.lin_src = MLP([hid_dim] + [hid_dim] * num_mlp_layers, act=act, norm=None, plain_last=False)
        self.lin_dst = MLP([hid_dim * 2] + [hid_dim] * num_mlp_layers, act=act, norm=None, plain_last=False)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), act=act, norm=None, plain_last=False)
        self.eps = torch.nn.Parameter(torch.Tensor([1.]))

    @torch.compile
    def forward(self, inputs, mask, data):
        # we need distinguish local and nonlocal 2-wl neighbors

        # inputs: B x N x N x F
        # mask: B x N x N
        B, N, _, _ = inputs.shape
        index = data.b.new_ones(data['vals'].num_nodes, 1) * -1.
        index[data.edge_index_dict[('obj', 'to', 'vals')][1]] = 1.
        if mask is not None:
            indicater = data.b.new_zeros(B, N, N, 1)
            indicater[mask] = index
        else:
            indicater = index.reshape(B, N, N, 1)

        indicated = torch.einsum('bnmd,bmld->bnld', inputs, indicater)
        indicated = indicated + indicated.transpose(1, 2)
        x = inputs + indicated

        x = self.lin_src(x)
        n = x.shape[1]
        if mask is not None:
            aggr_x = x.sum(1) / mask.sum(2, keepdim=True).float()  # B x N x F, B x N x 1
        else:
            aggr_x = x.mean(1)
        aggr_x = aggr_x.unsqueeze(1).repeat(1, n, 1, 1)  # B x N x N x F
        if mask is not None:
            aggr_x = aggr_x.masked_fill(~mask.unsqueeze(3), 0.)

        triu_mask = upper_triangle_mask(n, x.device)
        aggr_tuple = torch.cat([aggr_x, aggr_x.transpose(1, 2)], dim=-1)  # the 2WL tuple
        aggr_tuple = torch.where(triu_mask[None, :, :, None], aggr_tuple, aggr_tuple.transpose(1, 2))
        msg = self.lin_dst(aggr_tuple)
        x_dst = (1 + self.eps) * inputs + msg
        return self.mlp(x_dst)


class DeltaTwoWL(HigherOrder):
    def __init__(self,
                 no_mp,
                 no_wl,
                 no_dual,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 gnn_mlp_layers,
                 num_pred_layers,
                 block_mlp_layers,
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
            self.init_higher_order_layers(num_conv_layers, hid_dim, block_mlp_layers, act)

    def init_higher_order_layers(self, num_conv_layers, hid_dim, block_mlp_layers, act):
        self.higher_orders = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.higher_orders.append(GINEConv(hid_dim, block_mlp_layers, act))
