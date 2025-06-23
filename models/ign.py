# https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch/blob/master/layers/equivariant_linear_pytorch.py
import torch
import torch.nn as nn


# equi_2_to_1
class Layer2to1(nn.Module):
    """
    :param input_depth: D
    :param output_depth: S
    """
    def __init__(self, input_depth, output_depth, normalization='inf', normalization_val=1.0):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val

        self.basis_dimension = 1

        # initialization values for variables
        self.coeffs = nn.Parameter(
            torch.empty(self.input_depth, self.output_depth, self.basis_dimension), requires_grad=True)
        nn.init.xavier_normal_(self.coeffs)

        # bias
        self.bias = nn.Parameter(torch.zeros(1, self.output_depth, 1), requires_grad=True)

    def forward(self, inputs):
        """
        :param inputs: N x D x m x m tensor
        :return: output: N x S x m tensor
        """
        m = inputs.size(3)  # extract dimension

        ops_out = contractions_2_to_1(inputs, m, normalization=self.normalization)
        ops_out = torch.stack(ops_out, dim=2)  # N x D x B x m

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m

        # bias
        output = output + self.bias

        return output


# equi_1_to_2
class Layer1to2(nn.Module):
    """
    :param input_depth: D
    :param output_depth: S
    """

    def __init__(self, input_depth, output_depth, normalization='inf', normalization_val=1.0):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val

        self.basis_dimension = 1

        # initialization values for variables
        self.coeffs = nn.Parameter(
            torch.empty(self.input_depth, self.output_depth, self.basis_dimension), requires_grad=True)
        nn.init.xavier_normal_(self.coeffs)

        # bias
        self.bias = nn.Parameter(torch.zeros(1, self.output_depth, 1, 1))

    def forward(self, inputs):
        """
        :param inputs: N x D x m tensor
        :return: output: N x S x m x m tensor
        """
        m = inputs.size(2)  # extract dimension

        ops_out = contractions_1_to_2(inputs, m, normalization=self.normalization)
        ops_out = torch.stack(ops_out, dim=2)  # N x D x B x m x m

        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        # bias
        output = output + self.bias

        return output


# ops_2_to_1
def contractions_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    # diag_part = torch.diagonal(inputs, dim1=2, dim2=3)  # N x D x m

    # sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
    sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
    # sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
    # sum_all = torch.sum(inputs, dim=(2, 3))  # N x D

    # op1 - (123) - extract diag
    # op1 = diag_part  # N x D x m

    # op2 - (123) + (12)(3) - tile sum of diag part
    # op2 = sum_diag_part.repeat(1, 1, dim)
    # op2 = torch.cat([sum_diag_part for d in range(dim)], dim=2)  # N x D x m

    # op3 - (123) + (13)(2) - place sum of row i in element i
    op3 = sum_of_rows  # N x D x m

    # op4 - (123) + (23)(1) - place sum of col i in element i
    # op4 = sum_of_cols  # N x D x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    # op5 = sum_all.unsqueeze(dim=2).repeat(1, 1, dim)
    # op5 = torch.cat([sum_all.unsqueeze(dim=2) for d in range(dim)], dim=2)  # N x D x m

    if normalization is not None:
        if normalization == 'inf':
            # op2 = op2 / dim
            op3 = op3 / dim
            # op4 = op4 / dim
            # op5 = op5 / (dim ** 2)

    return [op3,]


# ops_1_to_2
def contractions_1_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    # sum_all = torch.sum(inputs, dim=2, keepdim=True)  # N x D x 1

    # op1 - (123) - place on diag
    # op1 = torch.diag_embed(inputs, dim1=2, dim2=3)  # N x D x m x m

    # op2 - (123) + (12)(3) - tile sum on diag
    # op2 = torch.diag_embed(sum_all.repeat(1, 1, dim), dim1=2, dim2=3)  # N x D x m x m

    # op3 - (123) + (13)(2) - tile element i in row i
    # op3 = inputs.unsqueeze(2).repeat(1, 1, dim, 1)
    # op3 = torch.cat([torch.unsqueeze(inputs, dim=2) for d in range(dim)], dim=2)  # N x D x m x m

    # op4 - (123) + (23)(1) - tile element i in col i
    # op4 = inputs.unsqueeze(3).repeat(1, 1, 1, dim)
    # op4 = torch.cat([torch.unsqueeze(inputs, dim=3) for d in range(dim)], dim=3)  # N x D x m x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    # op5 = sum_all.unsqueeze(3).repeat(1, 1, dim, dim)  # N x D x m x m

    # if normalization is not None:
    #     if normalization == 'inf':
            # op2 = op2 / dim
            # op5 = op5 / dim

    return [inputs.unsqueeze(2) + inputs.unsqueeze(3),]
