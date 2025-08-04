import torch
from torch_geometric.nn import MLP

from models.hetero_higher_order import HigherOrder
from models.util import upper_triangle_mask


# https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/blob/master/layers/modules.py
class PPGNBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, mlp_layers, layernorm, act):
        super().__init__()

        self.mlp1 = MLP([in_features] + [out_features] * mlp_layers, act=act, norm=None, plain_last=False)
        self.mlp2 = MLP([in_features] + [out_features] * mlp_layers, act=act, norm=None, plain_last=False)
        self.skip = MLP([out_features, out_features], act=act, norm=None, plain_last=False)
        if layernorm:
            self.ln = torch.nn.LayerNorm(out_features)
        else:
            self.ln = torch.nn.Identity()

    @torch.compile
    def forward(self, inputs, mask=None):
        x1 = self.mlp1(inputs)
        x2 = self.mlp2(inputs)
        if mask is not None:
            x1 = x1.masked_fill(~mask.unsqueeze(3), 0.)
            x2 = x2.masked_fill(~mask.unsqueeze(3), 0.)

        mult = torch.einsum('bmnf,bnlf->bmlf', x1, x2)
        mult = self.ln(mult)
        triu_mask = upper_triangle_mask(inputs.shape[1], x1.device)
        mult = torch.where(triu_mask[None, :, :, None], mult, mult.transpose(1, 2))

        # out = torch.cat([inputs, mult], dim=-1)
        out = self.skip(inputs + mult)
        return out


class PPGN(HigherOrder):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_gnn_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 block_mlp_layers,
                 layernorm,
                 norm,
                 act):
        super().__init__(hid_dim,
                         num_encode_layers,
                         num_gnn_layers,
                         num_conv_layers,
                         num_pred_layers,
                         num_mlp_layers,
                         norm,
                         act)

        self.higher_orders = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.higher_orders.append(PPGNBlock(hid_dim, hid_dim, block_mlp_layers, layernorm, act))
