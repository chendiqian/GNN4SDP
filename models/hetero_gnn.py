from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

from models.hetero_conv import HeteroConvLayer
from models.sageconv import SAGEConv
from models.ign import Layer2to1, Layer2to2


class GNN(torch.nn.Module):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act):
        super().__init__()

        self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.vals_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)

        self.gcns = torch.nn.ModuleList()
        self.igns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.igns.append(Layer2to2(hid_dim, hid_dim, act))
            self.gcns.append(HeteroConvLayer(
                v2c_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                c2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm)
            ))

        # Todo: potentially useful, to project 2d to 1d, then outer product for PSD prediction matrix
        self.ign_2to1 = Layer2to1(hid_dim, hid_dim, act)
        self.predictor = MLP([hid_dim] * num_pred_layers + [1], act=act, norm=None)

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        batch_dict['_vals'] = data.first_order_batch if hasattr(data, 'first_order_batch') else None
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.cons_encoder(data.b[:, None])
        vals_embedding = cons_embedding.new_zeros(data['vals'].num_nodes, 1)
        vals_embedding[edge_index_dict[('obj', 'to', 'vals')][1]] = edge_attr_dict[('obj', 'to', 'vals')]
        vals_embedding = self.vals_encoder(vals_embedding)

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding, 'cons': cons_embedding}
        return batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict

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

            x_x_dense = self.igns[i](x_x_dense.permute(0, 3, 1, 2))  # b x f x n x n
            x_x_dense = x_x_dense.permute(0, 2, 3, 1)[real_x_x_mask]

            x_dict['vals'] = x_x_dense  # sum(nnodes ** 2) x F
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        x = self.predictor(x_dict['vals']).squeeze()
        return x
