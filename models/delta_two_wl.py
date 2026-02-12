import torch
from torch_geometric.nn import MLP

from models.hetero_higher_order import HigherOrder
from models.util import upper_triangle_mask


class GINEConv(torch.nn.Module):
    def __init__(self, hid_dim, num_mlp_layers, act):
        super().__init__()

        self.lin_src = MLP([hid_dim] + [hid_dim] * num_mlp_layers, act=act, norm=None, plain_last=False)
        self.lin2_src = MLP([hid_dim] + [hid_dim] * num_mlp_layers, act=act, norm=None, plain_last=False)
        self.lin_dst = MLP([hid_dim] + [hid_dim] * num_mlp_layers, act=act, norm='layernorm', plain_last=False)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), act=act, norm=None, plain_last=False)
        self.eps = torch.nn.Parameter(torch.Tensor([1.]))

    @torch.compile
    def forward(self, inputs, mask, data):
        # we need distinguish local and nonlocal 2-wl neighbors

        # inputs: B x N x N x F
        # mask: B x N x N
        B, N, _, _ = inputs.shape

        # row, col aggr
        x = self.lin_src(inputs)
        if mask is not None:
            aggr_x = x.sum(1) / mask.sum(2, keepdim=True).float()  # B x N x F, B x N x 1
        else:
            aggr_x = x.mean(1)
        # repeat at dim=2, so that the elements on each row share the same
        aggr_x = aggr_x.unsqueeze(2).repeat(1, 1, N, 1)  # B x N x N x F
        if mask is not None:
            aggr_x = aggr_x.masked_fill(~mask.unsqueeze(3), 0.)

        # adj matmul aggr
        # todo: this can be more efficient
        index = data.b.new_zeros(data['vals'].num_nodes, 1)
        index[data.edge_index_dict[('obj', 'to', 'vals')][1]] = data.edge_attr_dict[('obj', 'to', 'vals')]

        # todo: try to encode this as well
        if mask is not None:
            indicater = data.b.new_zeros(B, N, N, 1)
            indicater[mask] = index
        else:
            indicater = index.reshape(B, N, N, 1)

        x = self.lin2_src(inputs)
        # H @ adj
        # todo: N ** 0.5
        indicated = torch.einsum('bnmd,bmld->bnld', x, indicater) / (N ** 0.5)

        # todo: try add
        embedding = aggr_x + indicated

        msg = self.lin_dst(embedding)
        msg = msg + msg.transpose(1, 2)
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
