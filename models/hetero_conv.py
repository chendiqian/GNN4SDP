from typing import Dict
from typing import Optional

import torch
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn.conv.hetero_conv import group


class HeteroConv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        raise NotImplementedError

    def forward(
            self,
            x_dict: Dict[NodeType, torch.FloatTensor],
            batch_dict: Dict[NodeType, torch.LongTensor],
            edge_index_dict: Dict[EdgeType, torch.LongTensor],
            edge_attr_dict: Dict[EdgeType, torch.FloatTensor]
    ) -> Dict[NodeType, torch.FloatTensor]:

        new_dict = {}
        for conv_group in self.conv_sequence:
            current_results = []
            dst = conv_group[0].split('_')[1]
            for conv in conv_group:
                src, dst = conv.split('_')
                args = [(x_dict[src], x_dict[dst])]
                args = args + [edge_index_dict[(src, 'to', dst)],
                               edge_attr_dict[(src, 'to', dst)],
                               batch_dict[dst]]

                current_results.append(self.convs[conv](*args))

            if self.sync_conv:
                new_dict[dst] = group(current_results, 'mean')
            else:
                x_dict[dst] = group(current_results, 'mean')

        return new_dict if self.sync_conv else x_dict


class HeteroConvLayer(HeteroConv):
    def __init__(
            self,
            v2c_conv: torch.nn.Module,
            c2v_conv: torch.nn.Module,
            sync_conv: bool = False
    ):
        super().__init__()

        self.convs = torch.nn.ModuleDict(
            {'vals_cons': v2c_conv,
             'cons_vals': c2v_conv}
        )
        # we use c -> v -> o setting, so o is the final output
        self.conv_sequence = [('vals_cons',),
                              ('cons_vals',)]
        self.sync_conv = sync_conv

    def reset_parameters(self):
        self.convs['vals_cons'].reset_parameters()
        self.convs['cons_vals'].reset_parameters()
