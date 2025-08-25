from typing import Dict, Optional

import torch
from torch_geometric.nn import MessagePassing, MLP, Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

from models.hetero_conv import HeteroConvLayer
from models.util import need_padding


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


class MPNN(torch.nn.Module):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_gnn_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act):
        super().__init__()

        self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.vals_encoder = MLP([2] + [hid_dim] * num_encode_layers, act=act, norm=None)

        self.gcns = torch.nn.ModuleList()
        assert num_gnn_layers > 0
        for layer in range(num_gnn_layers):
            self.gcns.append(HeteroConvLayer(
                v2c_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                c2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
            ))

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], act=act, norm=None)

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        batch_dict['_vals'] = data.first_order_batch if hasattr(data, 'first_order_batch') else None
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict

        cons_embedding = self.cons_encoder(data.b[:, None])

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

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding, 'cons': cons_embedding}
        return batch_dict, edge_index_dict, edge_attr_dict, x_dict

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, x_dict = self.init_embedding(data)

        for i, layer in enumerate(self.gcns):
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict)

        return self.predictor(x_dict['vals']).squeeze()
