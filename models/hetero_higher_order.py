from typing import Dict, Tuple

import torch
from torch_geometric.nn import MLP, MessagePassing, Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

from models.util import need_padding
from models.ign_21 import Layer2to1


class SpatialLayerNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, 1, num_features))  # shape: (1, 1, 1, F)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, 1, num_features))

    def forward(self, x):
        # x shape: (B, N, N, F)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta


class HeteroConvLayer(torch.nn.Module):
    def __init__(
            self,
            v2c_conv: torch.nn.Module,
            c2v_conv: torch.nn.Module,
    ):
        super().__init__()

        self.vals_cons = v2c_conv
        self.cons_vals = c2v_conv
        self.eps = torch.nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(
            self,
            cons, vals,
            batch_dict: Dict[NodeType, torch.LongTensor],
            edge_index_dict: Dict[EdgeType, torch.LongTensor],
            edge_attr_dict: Dict[EdgeType, torch.FloatTensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        updated_cons = self.vals_cons(
            (vals, cons),
            edge_index_dict[('vals', 'to', 'cons')],
            edge_attr_dict[('vals', 'to', 'cons')],
            batch_dict['cons'])

        updated_vals = self.cons_vals(
            (updated_cons, vals),
            edge_index_dict[('cons', 'to', 'vals')],
            edge_attr_dict[('cons', 'to', 'vals')],
            batch_dict['vals']) * self.eps

        return updated_vals, updated_cons


class SAGEConv(MessagePassing):
    def __init__(self, hid_dim, num_mlp_layers, act, norm):
        super(SAGEConv, self).__init__(aggr='add')

        self.act = activation_resolver(act)
        self.lin_src = Linear(hid_dim, hid_dim)
        self.lin_dst = Linear(hid_dim, hid_dim)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), act=act, norm=norm, plain_last=False)

    def reset_parameters(self):
        self.lin_dst.reset_parameters()
        self.lin_src.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        x = (self.lin_src(x[0]), x[1])
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        x_dst = x[1]
        x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        return self.mlp(out, batch)

    def message(self, x_j, edge_attr):
        return self.act(x_j) * edge_attr

    def update(self, aggr_out):
        return aggr_out


class HigherOrder(torch.nn.Module):
    def __init__(self,
                 no_mp,
                 no_wl,
                 no_dual,
                 hid_dim,
                 out_dim,
                 num_encode_layers,
                 num_conv_layers,
                 gnn_mlp_layers,
                 num_pred_layers,
                 norm,
                 act):
        super().__init__()
        self.vals_encoder = MLP([2] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.num_conv_layers = num_conv_layers

        self.gcns = None
        self.cons_encoder = None
        if not no_mp:
            # we need constraint node only if we have message passing
            # in some cases e.g. maxcut, we may not need that
            self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
            self.gcns = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.gcns.append(HeteroConvLayer(
                    v2c_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=gnn_mlp_layers, act=act, norm=norm),
                    c2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=gnn_mlp_layers, act=act, norm=norm),
                ))

        self.norms = None
        self.higher_orders = None
        # the higher order models will be initialized in subclasses!!
        if not no_wl:
            self.init_higher_order_norms(num_conv_layers, hid_dim)

        # self.predictor2 = Layer2to1(hid_dim, 1, num_pred_layers, act)
        self.predictor = MLP([hid_dim] * num_pred_layers + [1], act=act, norm=None)

    def init_higher_order_layers(self, *args, **kwargs):
        raise NotImplementedError

    def init_higher_order_norms(self, num_conv_layers, hid_dim):
        self.norms = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.norms.append(SpatialLayerNorm(hid_dim))

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        batch_dict['_vals'] = data.first_order_batch if hasattr(data, 'first_order_batch') else None
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict

        cons_embedding = None
        if self.cons_encoder:
            cons_embedding = self.cons_encoder(data.b[:, None])

        # reshape encoded SDP into batch of square features
        if need_padding(batch_dict['_vals']):
            _, real_x_mask = to_dense_batch(data.b.new_empty(batch_dict['_vals'].shape[0]),
                                            batch_dict['_vals'])  # B x Nmax x F
            real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax
            B = real_x_x_mask.shape[0]
            N = real_x_x_mask.shape[1]
        else:
            real_x_x_mask = None
            B = batch_dict['_vals'].max() + 1
            N = batch_dict['_vals'].shape[0] // B

        vals_encoding = data.b.new_zeros(data['vals'].num_nodes, 1)
        vals_encoding[edge_index_dict[('obj', 'to', 'vals')][1]] = edge_attr_dict[('obj', 'to', 'vals')]
        # encode the diagonal entries
        diag_enc = torch.eye(N, dtype=torch.float, device=data.b.device)[None].repeat(B, 1, 1)
        if real_x_x_mask is not None:
            diag_enc = diag_enc[real_x_x_mask][..., None]
        else:
            diag_enc = diag_enc.reshape(-1, 1)
        vals_embedding = torch.hstack([diag_enc, vals_encoding])
        vals_embedding = self.vals_encoder(vals_embedding)

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding, 'cons': cons_embedding}
        return batch_dict, edge_index_dict, edge_attr_dict, x_dict, real_x_x_mask, vals_encoding.squeeze(1)

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, x_dict, real_x_x_mask, vals_encoding = self.init_embedding(data)
        if real_x_x_mask is not None:
            B = real_x_x_mask.shape[0]
            N = real_x_x_mask.shape[1]
        else:
            B = batch_dict['_vals'].max() + 1
            N = batch_dict['_vals'].shape[0] // B

        # init vals is flat!
        cons, vals = x_dict['cons'], x_dict['vals']

        for i in range(self.num_conv_layers):
            # WL
            if self.higher_orders and self.norms:
                # need to turn them into N x N x F shape first
                if self.gcns or i == 0:
                    if real_x_x_mask is not None:
                        x_x_dense = torch.zeros(*real_x_x_mask.shape + (vals.shape[-1],), device=vals.device, dtype=torch.float)
                        x_x_dense[real_x_x_mask] = vals
                        vals = x_x_dense
                    else:
                        vals = vals.reshape(B, N, N, -1)

                # update
                vals = self.higher_orders[i](vals, real_x_x_mask)
                vals = self.norms[i](vals)

                # flatten them for message passing or final prediction
                if self.gcns or i == self.num_conv_layers - 1:
                    if real_x_x_mask is not None:
                        vals = vals[real_x_x_mask]
                    else:
                        vals = vals.reshape(-1, vals.shape[-1])

            # mpnn
            if self.gcns:
                vals, cons = self.gcns[i](cons, vals, batch_dict, edge_index_dict, edge_attr_dict)

        # pred_s_latent = self.predictor(vals).squeeze(-1)

        # L, Q = torch.linalg.eigh(pred_s_latent)
        # L_new = torch.clamp(L, min=0.)
        # pred_S = (Q * L_new[:, None, :]) @ Q.transpose(-1, -2)

        # pred_S = torch.einsum('bnf,bmf->bnm', pred_s_latent, pred_s_latent)
        pred_y = self.predictor(cons).reshape(B, N)

        batched_C = vals_encoding.reshape(B, N, N)

        pred_S = batched_C - torch.diag_embed(pred_y, dim1=1, dim2=2)
        eigvals = torch.linalg.eigvalsh(pred_S)
        delta = torch.min(eigvals, dim=1).values
        delta = torch.clamp(-delta, min=0.)
        pred_y = pred_y - delta[:, None]
        pred_S = batched_C - torch.diag_embed(pred_y, dim1=1, dim2=2)

        return pred_y, pred_S, batched_C
