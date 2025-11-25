import torch
from torch_geometric.nn import MLP

from models.hetero_higher_order import HigherOrder


class TwoFWLBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, mlp_layers, act):
        super().__init__()
        self.mlp1 = MLP([in_features] + [out_features] * mlp_layers, act=act, norm=None, plain_last=False)
        self.mlp2 = MLP([out_features * 2] + [out_features] * mlp_layers, act=act, norm=None, plain_last=False)
        self.skip = MLP([out_features] * (mlp_layers + 1), act=act, norm=None, plain_last=False)

    # @torch.compile
    def forward(self, inputs, mask=None):
        # b * n * n * f
        # b * n * n
        B, N, _, F = inputs.shape
        X = self.mlp1(inputs)

        if mask is not None:
            X = X.masked_fill(~mask.unsqueeze(3), 0.)

        row, col = torch.triu_indices(N, N, 0, device=X.device)
        #  b * (n+1)n/2 * n * f
        Xnew = torch.cat([X[:, row, :, :], X[:, col, :, :]], dim=-1)
        # aggregate and normalize
        Xnew = self.mlp2(Xnew)
        Xnew = Xnew.sum(2) / N

        # (n+1)n/2 * f * b
        Xnew = Xnew.permute((1, 2, 0))

        x_x_dense = torch.zeros(N, N, F, B, device=X.device, dtype=torch.float)
        x_x_dense[row, col] = Xnew
        x_x_dense = x_x_dense + x_x_dense.transpose(0, 1)

        x_x_dense = x_x_dense.permute((3, 0, 1, 2))

        out = self.skip(X + x_x_dense)
        return out


class TwoFWL(HigherOrder):
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
            self.higher_orders.append(TwoFWLBlock(hid_dim, hid_dim, block_mlp_layers, act))
