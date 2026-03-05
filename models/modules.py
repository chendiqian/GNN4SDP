from typing import Dict, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, Linear, MLP
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import NodeType, EdgeType


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


class Layer2to1(nn.Module):
    def __init__(self, input_depth, output_depth, mlp_layers, act):
        super().__init__()

        self.mlp = MLP([input_depth] * mlp_layers + [input_depth], act=act, norm=None)

        self.score_net = MLP([input_depth, input_depth, 1], act='tanh', norm=None)

        self.out_proj = nn.Linear(input_depth, output_depth)

    def forward(self, pairwise_embeddings):
        # input is 4D (Batch, N, N, F)
        pairwise_embeddings = self.mlp(pairwise_embeddings)

        B, N, _, F_dim = pairwise_embeddings.shape

        scores = self.score_net(pairwise_embeddings)
        alpha = F.softmax(scores, dim=2)
        weighted_features = alpha * pairwise_embeddings

        L_node = weighted_features.sum(dim=2)

        L_out = self.out_proj(L_node)
        return L_out
