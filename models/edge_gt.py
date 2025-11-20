import torch

from models.hetero_higher_order import HigherOrder
from models.util import upper_triangle_mask


# https://github.com/luis-mueller/towards-principled-gts/blob/main/edge_transformer.py#L465
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
    def forward(self, inputs, mask=None):
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
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(4), -1e9)

        att = torch.softmax(scores, dim=2)
        val = left_v.unsqueeze(1) * right_v.unsqueeze(3)  # bnmlhf

        x = torch.einsum('bnmlh,bnmlhf->bnlhf', att, val)
        x = x.view(B, N, N, F)

        triu_mask = upper_triangle_mask(N, val.device)
        x = torch.where(triu_mask[None, :, :, None], x, x.transpose(1, 2))
        return self.olin(x + inputs)


class EdgeGT(HigherOrder):
    def __init__(self,
                 no_mp,
                 no_wl,
                 no_dual,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 gnn_mlp_layers,
                 num_pred_layers,
                 num_head,
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
            self.init_higher_order_layers(num_conv_layers, hid_dim, num_head)

    def init_higher_order_layers(self, num_conv_layers, hid_dim, num_head):
        self.higher_orders = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.higher_orders.append(FastEdgeAttention(hid_dim, num_head))
