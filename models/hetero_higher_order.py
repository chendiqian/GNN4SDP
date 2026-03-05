from typing import Dict

import torch
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

from models.modules import SpatialLayerNorm, HeteroConvLayer, SAGEConv, Layer2to1
from models.util import need_padding


class HigherOrder(torch.nn.Module):
    def __init__(self,
                 no_mp,
                 no_wl,
                 no_dual,
                 target,
                 hid_dim,
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

        self.target = target
        assert target in ['primal', 'dual', 'primal+dual']
        self.primal_predictor = None
        self.dual_predictor = None
        if 'dual' in target:
            # predict dual y
            self.dual_predictor = Layer2to1(hid_dim, 1, num_pred_layers, act)
        if 'primal' in target:
            # predict latent X
            self.primal_predictor = Layer2to1(hid_dim, hid_dim, num_pred_layers, act)

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

        pred_X = None
        pred_y = None
        pred_S = None

        # batch C and b, for objective evaluation
        if real_x_x_mask is not None:
            C = torch.zeros(*real_x_x_mask.shape, device=vals.device, dtype=torch.float)
            C[real_x_x_mask] = vals_encoding

            vals_dense = torch.zeros(*real_x_x_mask.shape + (vals.shape[-1],), device=vals.device, dtype=torch.float)
            vals_dense[real_x_x_mask] = vals
        else:
            C = vals_encoding.reshape(B, N, N)
            vals_dense = vals.reshape(B, N, N, -1)

        if need_padding(batch_dict['cons']):
            batch_b, real_y_mask = to_dense_batch(data.b, batch_dict['cons'])
        else:
            real_y_mask = None
            batch_b = data.b.reshape(B, -1)

        if 'dual' in self.target:
            pred_y = self.dual_predictor(vals_dense).squeeze(-1)
            if real_y_mask is not None:
                # we need to mask out some paddings!
                pred_y = pred_y.masked_fill(~real_y_mask, 0.)
            pred_S = C - torch.diag_embed(pred_y, dim1=1, dim2=2)
            # todo: might be able to improved
            eigvals = torch.linalg.eigvalsh(pred_S)
            delta = torch.min(eigvals, dim=1).values
            delta = torch.clamp(-delta, min=0.)
            # corrected pred y, that satisfies the PSD of S
            pred_y = pred_y - delta[:, None]
            if real_y_mask is not None:
                # we need to mask out some paddings!
                pred_y = pred_y.masked_fill(~real_y_mask, 0.)
            # pred_S is already masked for new paddings
            pred_S = C - torch.diag_embed(pred_y, dim1=1, dim2=2)

        if 'primal' in self.target:
            pred_X_latent = self.primal_predictor(vals_dense)
            pred_X_latent = torch.nn.functional.normalize(pred_X_latent, p=2, dim=2)
            pred_X = torch.einsum('bnf,bmf->bnm', pred_X_latent, pred_X_latent)
            if real_x_x_mask is not None:
                pred_X = pred_X.masked_fill(~real_x_x_mask, 0.)

        return pred_X, pred_y, pred_S, C, batch_b
