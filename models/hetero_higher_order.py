from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP, global_add_pool
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

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
                 act):
        super().__init__()

        self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.vals_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)

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
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        if self.encode_type == 'gnn':
            cons_embedding = self.cons_encoder(data.b[:, None])
            vals_embedding = cons_embedding.new_zeros(data['vals'].num_nodes, 1)
            vals_embedding[edge_index_dict[('obj', 'to', 'vals')][1]] = edge_attr_dict[('obj', 'to', 'vals')]
            vals_embedding = self.vals_encoder(vals_embedding)

            x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding, 'cons': cons_embedding}
            for i, layer in enumerate(self.gcns):
                x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)
            x = x_dict['vals']
        elif self.encode_type == 'multiset':
            # cat (A_ij, b) first
            b = data.b[edge_index_dict[('cons', 'to', 'vals')][0]].unsqueeze(1)
            cons = torch.cat([edge_attr_dict[('cons', 'to', 'vals')], b], dim=1)
            # encode (A_ij, b) pairs
            cons = self.cons_encoder(cons)
            # aggregate to variable nodes
            cons = global_add_pool(cons, edge_index_dict[('cons', 'to', 'vals')][1], data['vals'].num_nodes)

            vals_embedding = cons.new_zeros(data['vals'].num_nodes, 1)
            vals_embedding[edge_index_dict[('obj', 'to', 'vals')][1]] = edge_attr_dict[('obj', 'to', 'vals')]
            vals_embedding = self.vals_encoder(vals_embedding)

            x = self.encoder(torch.cat([vals_embedding, cons], dim=1), batch_dict['vals'])
        else:
            raise NotImplementedError

        return batch_dict, x

    def forward(self, data):
        batch_dict, x = self.init_embedding(data)

        feature_dim = x.shape[-1]
        device = x.device

        # reshape encoded SDP into batch of square features
        if need_padding(batch_dict['_vals']):
            _, real_x_mask = to_dense_batch(x.new_empty(batch_dict['_vals'].shape[0]),
                                            batch_dict['_vals'])  # B x Nmax x F
            real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax

            x_x_dense = torch.zeros(*real_x_x_mask.shape + (feature_dim,), device=device, dtype=torch.float)
            x_x_dense[real_x_x_mask] = x
        else:
            real_x_x_mask = None
            B = batch_dict['_vals'].max() + 1
            N = batch_dict['_vals'].shape[0] // B
            x_x_dense = x.reshape(B, N, N, -1)

        # higher order layers
        for i, layer in enumerate(self.higher_orders):
            x_x_dense = self.higher_orders[i](x_x_dense, real_x_x_mask)
            x_x_dense = self.norms[i](x_x_dense)

        # readout
        if real_x_x_mask is not None:
            x_x_dense = x_x_dense[real_x_x_mask]
        else:
            x_x_dense = x_x_dense.reshape(-1, feature_dim)
        return self.predictor(x_x_dense).squeeze()

    def predict_single(self, data):
        raise NotImplementedError
