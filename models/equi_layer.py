import torch.nn as nn
from torch_geometric.nn import MLP


# equi_2_to_1
class EquiLayer2to1(nn.Module):
    def __init__(self, input_depth, output_depth, mlp_layers, act, norm):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.mlp = MLP([input_depth] + [output_depth] * (mlp_layers + 1), act=act, norm=norm, plain_last=False)

    def forward(self, inputs, real_mask, batch):
        # B x Nmax x Nmax x F
        return self.mlp(inputs.mean(1, keepdim=False)[real_mask], batch)


# equi_1_to_2
class EquiLayer1to2(nn.Module):
    """
    :param input_depth: D
    :param output_depth: S
    """

    def __init__(self, input_depth, output_depth, mlp_layers, act, norm):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.mlp = MLP([input_depth] + [output_depth] * (mlp_layers + 1), act=act, norm=norm, plain_last=False)

    def forward(self, inputs, real_mask, batch):
        # B x Nmax x F
        inputs = inputs.unsqueeze(1) + inputs.unsqueeze(2)
        return self.mlp(inputs[real_mask], batch)
