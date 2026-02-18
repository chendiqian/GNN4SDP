from torch_geometric.nn import MLP
import torch.nn as nn
import torch.nn.functional as F


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
