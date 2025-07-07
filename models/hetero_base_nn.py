from typing import Dict, Optional

import torch
from torch_geometric.nn import MessagePassing, MLP, Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import EdgeType, NodeType

from models.hetero_conv import HeteroConvLayer


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


class Layer2to1(torch.nn.Module):
    def __init__(self, input_depth, output_depth, act):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.act = activation_resolver(act)
        self.basis_dimension = 2

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(
            torch.empty(self.basis_dimension, self.input_depth, self.output_depth), requires_grad=True)
        torch.nn.init.xavier_normal_(self.coeffs)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, self.output_depth), requires_grad=True)

    def forward(self, inputs):
        """
        :param inputs: N x m x m x D tensor
        :return: output: N x m x S tensor
        """
        t1 = torch.diagonal(inputs, dim1=1, dim2=2).transpose(1, 2)  # N x m x D
        t2 = torch.mean(inputs, dim=1)  # N x m x D
        inputs = torch.stack([t1, t2], dim=0)  # op x N x m x D

        output = torch.einsum('dfh,dbmf->bmh', self.coeffs, inputs)  # N x S x m

        # bias
        output = output + self.bias

        return self.act(output)


class BaseModel(torch.nn.Module):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act,
                 force_psd):
        super().__init__()

        self.cons_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)
        self.vals_encoder = MLP([1] + [hid_dim] * num_encode_layers, act=act, norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(HeteroConvLayer(
                v2c_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
                c2v_conv=SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, act=act, norm=norm),
            ))

        # potentially useful, to project 2d to 1d, then outer product for PSD prediction matrix
        self.ign_2to1 = Layer2to1(hid_dim, hid_dim, act) if force_psd else None
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
        raise NotImplementedError
