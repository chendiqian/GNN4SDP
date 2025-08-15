from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP, global_add_pool
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch, unbatch_edge_index

from models.hetero_base_nn import SAGEConv
from models.hetero_conv import HeteroConvLayer
from models.util import need_padding


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


class HigherOrder(torch.nn.Module):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 encode_type,
                 num_gnn_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act,
                 max_con_nodes=None,
                 max_val_nodes=None):
        super().__init__()

        self.max_con_nodes = max_con_nodes
        self.max_val_nodes = max_val_nodes

        self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.vals_encoder = MLP([2] + [hid_dim] * num_encode_layers, act=act, norm=None)

        self.encode_type = encode_type
        if encode_type == 'gnn':
            self.gcns = torch.nn.ModuleList()
            assert num_gnn_layers > 0
            for layer in range(num_gnn_layers):
                self.gcns.append(HeteroConvLayer(
                    v2c_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                    c2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                ))
        elif encode_type == 'multiset':
            # overwrite
            self.cons_encoder = MLP([2] + [hid_dim] * num_encode_layers, act=act, norm=None)
            # use param num_gnn_layers as number of encoder MLP
            self.encoder = MLP([hid_dim * 2] + [hid_dim] * num_gnn_layers, act=act, norm=norm,
                               plain_last=False)
        elif encode_type == 'cat':
            self.cons_encoder = MLP([2 * max_con_nodes] + [hid_dim] * num_encode_layers, act=act, norm=None)
            self.encoder = MLP([hid_dim * 2] + [hid_dim] * num_gnn_layers, act=act, norm=norm,
                               plain_last=False)

        self.norms = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.norms.append(SpatialLayerNorm(hid_dim))

        # higher order NN is defined in separate instantiations!

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], act=act, norm=None)

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        batch_dict['_vals'] = data.first_order_batch if hasattr(data, 'first_order_batch') else None
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict

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

        # encode the diagonal entries
        diag_enc = torch.eye(N, dtype=torch.float, device=data.b.device)[None].repeat(B, 1, 1)
        if real_x_x_mask is not None:
            diag_enc = diag_enc[real_x_x_mask]
        else:
            diag_enc = diag_enc.reshape(-1, 1)

        vals_embedding = data.b.new_zeros(data['vals'].num_nodes, 1)
        vals_embedding[edge_index_dict[('obj', 'to', 'vals')][1]] = edge_attr_dict[('obj', 'to', 'vals')]
        vals_embedding = torch.hstack([diag_enc, vals_embedding])
        vals_embedding = self.vals_encoder(vals_embedding)

        if self.encode_type == 'gnn':
            cons_embedding = self.cons_encoder(data.b[:, None])

            x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding, 'cons': cons_embedding}
            for i, layer in enumerate(self.gcns):
                x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict)
            x = x_dict['vals']
        elif self.encode_type == 'multiset':
            # cat (A_ij, b) first
            b = data.b[edge_index_dict[('cons', 'to', 'vals')][0]].unsqueeze(1)
            cons = torch.cat([edge_attr_dict[('cons', 'to', 'vals')], b], dim=1)
            # encode (A_ij, b) pairs
            cons = self.cons_encoder(cons)
            # aggregate to variable nodes
            cons = global_add_pool(cons, edge_index_dict[('cons', 'to', 'vals')][1], data['vals'].num_nodes)

            x = self.encoder(torch.cat([vals_embedding, cons], dim=1), batch_dict['vals'])
        elif self.encode_type == 'cat':
            dense_constraints = vals_embedding.new_zeros(data['vals'].num_nodes, self.max_con_nodes)
            val_idx = edge_index_dict[('cons', 'to', 'vals')][1]
            con_idx = torch.hstack(unbatch_edge_index(edge_index_dict[('cons', 'to', 'vals')][:1],
                                                      batch_dict['cons']))[0]
            dense_constraints[val_idx, con_idx] = edge_attr_dict[('cons', 'to', 'vals')].squeeze(1)
            dense_b, _ = to_dense_batch(data.b, batch_dict['cons'], max_num_nodes=self.max_con_nodes)
            nnodes_vals = torch.unique(batch_dict['vals'], return_counts=True)[1]
            dense_b = dense_b.repeat_interleave(nnodes_vals, dim=0)
            cons = torch.hstack([dense_b, dense_constraints])
            cons_embedding = self.cons_encoder(cons)
            x = self.encoder(torch.cat([vals_embedding, cons_embedding], dim=1), batch_dict['vals'])
        else:
            raise NotImplementedError

        if real_x_x_mask is not None:
            x_x_dense = torch.zeros(*real_x_x_mask.shape + (x.shape[-1],), device=x.device, dtype=torch.float)
            x_x_dense[real_x_x_mask] = x
        else:
            x_x_dense = x.reshape(B, N, N, -1)

        return x_x_dense, real_x_x_mask

    def forward(self, data):
        x_x_dense, real_x_x_mask = self.init_embedding(data)

        # higher order layers
        for i, layer in enumerate(self.higher_orders):
            x_x_dense = self.higher_orders[i](x_x_dense, real_x_x_mask)
            x_x_dense = self.norms[i](x_x_dense)

        # readout
        if real_x_x_mask is not None:
            x_x_dense = x_x_dense[real_x_x_mask]
        else:
            x_x_dense = x_x_dense.reshape(-1, x_x_dense.shape[-1])
        return self.predictor(x_x_dense).squeeze()

    def predict_single(self, data):
        raise NotImplementedError
