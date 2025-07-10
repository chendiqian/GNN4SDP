import pdb

import torch
from torch_geometric.utils import to_dense_batch

from models.hetero_base_nn import BaseModel
from models.util import upper_triangle_mask


class FastEdgeAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.qlin = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.klin = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v1lin = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v2lin = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.olin = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    @torch.compile
    def forward(self, inputs, mask):
        # B N N F
        B, N, _, F = inputs.shape

        left_k = self.qlin(inputs)
        right_k = self.klin(inputs)
        left_v = self.v1lin(inputs)
        right_v = self.v2lin(inputs)

        left_k = left_k.view(
            B, N, N, self.num_heads, self.d_k
        )
        right_k = right_k.view_as(left_k)
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        scores = torch.einsum('bnmhf,bmlhf->bnmlh', left_k, right_k) / self.d_k ** 0.5
        scores = scores.masked_fill(~mask.unsqueeze(4), -1e9)

        att = torch.softmax(scores, dim=2)
        val = left_v.unsqueeze(1) * right_v.unsqueeze(3)  # bnmlhf
        mask = upper_triangle_mask(N, val.device)
        val = torch.where(mask[None, :, None, :, None, None], val, val.transpose(1, 3))

        x = torch.einsum('bnmlh,bnmlhf->bnlhf', att, val)
        x = x.view(B, N, N, F)
        return self.olin(x + inputs)


class EdgeGT(BaseModel):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 num_head,
                 norm,
                 act):
        super().__init__(hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         num_mlp_layers,
                         norm,
                         act)

        self.gt_layers = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gt_layers.append(FastEdgeAttention(hid_dim, num_head))

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        _, real_x_mask = to_dense_batch(x_dict['vals'].new_empty(batch_dict['_vals'].shape[0]),
                                        batch_dict['_vals'])  # B x Nmax x F
        real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax
        real_x_x_x_mask = torch.einsum('bnm,bml->bnml', real_x_x_mask, real_x_x_mask)
        feature_dim = x_dict['vals'].shape[-1]
        device = x_dict['vals'].device

        for i, layer in enumerate(self.gcns):
            x_x_dense = torch.zeros(*real_x_x_mask.shape + (feature_dim,), device=device, dtype=torch.float)
            x_x_dense[real_x_x_mask] = x_dict['vals']

            x_x_dense = self.gt_layers[i](x_x_dense, real_x_x_x_mask)
            x_x_dense = x_x_dense[real_x_x_mask]

            x_dict['vals'] = x_x_dense  # sum(nnodes ** 2) x F
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        return self.predictor(x_dict['vals']).squeeze()
