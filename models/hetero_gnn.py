from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch
# from torch_geometric.nn.resolver import activation_resolver

from models.hetero_conv import HeteroConvLayer
from models.sageconv import SAGEConv
# from models.ign import Layer1to2, Layer2to1
from models.equi_layer import EquiLayer1to2, EquiLayer2to1


class GNN(torch.nn.Module):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 ign_mlp_layer,
                 norm,
                 act):
        super().__init__()

        self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.vals_encoder = MLP([2] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.obj_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)

        self.gcns = torch.nn.ModuleList()
        self.ign_1to2 = torch.nn.ModuleList()
        self.ign_2to1 = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.ign_1to2.append(EquiLayer1to2(hid_dim, hid_dim, ign_mlp_layer, act, norm))

            if layer != num_conv_layers - 1:
                self.ign_2to1.append(EquiLayer2to1(hid_dim, hid_dim, ign_mlp_layer, act, norm))

            self.gcns.append(HeteroConvLayer(
                v2c_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                c2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                v2o_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm='instance_norm'),
                o2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
            ))

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], act=act, norm=None)

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        batch_dict['_vals'] = data.first_order_batch if hasattr(data, 'first_order_batch') else None
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.cons_encoder(data.b[:, None])
        obj_embedding = self.obj_encoder(cons_embedding.new_ones(data['obj'].num_nodes, 1))

        # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/laplace_pos_encoder.py#L99
        pos_enc = torch.stack((data.c_eigvec, data.c_eigval), dim=-1)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = self.vals_encoder(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        vals_embedding = torch.mean(pos_enc, 1, keepdim=False)

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                     'cons': cons_embedding,
                                                     'obj': obj_embedding}
        return batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        real_x_x_mask = None
        for i, layer in enumerate(self.gcns):
            x_dense, real_x_mask = to_dense_batch(x_dict['vals'], batch_dict['_vals'])  # B x Nmax x F
            # calculate only once
            if real_x_x_mask is None:
                real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax

            x_x_dense = self.ign_1to2[i](x_dense, real_x_x_mask, batch_dict['vals'])
            x_dict['vals'] = x_x_dense  # sum(nnodes ** 2) x F
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

            if i != len(self.gcns) - 1:  # not the last layer
                new_x_x_dense = x_x_dense.new_zeros(*real_x_x_mask.shape + (x_x_dense.shape[-1],))
                new_x_x_dense[real_x_x_mask] = x_dict['vals']  # B x Nmax x Nmax x F
                new_x = self.ign_2to1[i](new_x_x_dense, real_x_mask, batch_dict['_vals'])
                x_dict['vals'] = new_x

        x = self.predictor(x_dict['vals']).squeeze()
        return x
