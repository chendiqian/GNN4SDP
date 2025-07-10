import torch
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch

from models.hetero_base_nn import BaseModel
from models.util import upper_triangle_mask


# https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/blob/master/layers/modules.py
class PPGNBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, mlp_layers, act):
        super().__init__()

        self.mlp1 = MLP([in_features] + [out_features] * mlp_layers, act=act, norm=None, plain_last=False)
        self.mlp2 = MLP([in_features] + [out_features] * mlp_layers, act=act, norm=None, plain_last=False)
        self.skip = MLP([out_features, out_features], act=act, norm=None, plain_last=False)

    @torch.compile
    def forward(self, inputs, mask):
        x1 = self.mlp1(inputs).masked_fill(~mask.unsqueeze(3), 0.)
        x2 = self.mlp2(inputs).masked_fill(~mask.unsqueeze(3), 0.)

        mult = torch.einsum('bmnf,bnlf->bmlf', x1, x2)
        triu_mask = upper_triangle_mask(inputs.shape[1], x1.device)
        mult = torch.where(triu_mask[None, :, :, None], mult, mult.transpose(1, 2))

        # out = torch.cat([inputs, mult], dim=-1)
        out = self.skip(inputs + mult)
        return out


class PPGN(BaseModel):
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

        self.ppgns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.ppgns.append(PPGNBlock(hid_dim, hid_dim, block_mlp_layers, act))

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

            x_x_dense = self.ppgns[i](x_x_dense, real_x_x_mask)
            x_x_dense = x_x_dense[real_x_x_mask]

            x_dict['vals'] = x_x_dense
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        return self.predictor(x_dict['vals']).squeeze()
