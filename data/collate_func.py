from typing import List

import torch
from torch_geometric.data import Data, Batch


def collate_fn_lp_base(graphs: List[Data]):
    new_batch = Batch.from_data_list(graphs, exclude_keys=['x', 'nulls', 'c_eigval', 'c_eigvec'])
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    for k, v in graphs[0].edge_index_dict.items():
        src, rel, dst = k
        if src != dst:
            new_batch[(dst, rel, src)].edge_index = new_batch[(src, rel, dst)].edge_index[flip_tensor]
            new_batch[(dst, rel, src)].edge_attr = new_batch[(src, rel, dst)].edge_attr

    if not hasattr(new_batch[('cons', 'to', 'vals')], 'norm'):
        norm_dict = {}
        for k, v in new_batch.edge_index_dict.items():
            norm_dict[k] = None
        new_batch.norm_dict = norm_dict

    val_nnodes = new_batch['vals'].ptr[1:] - new_batch['vals'].ptr[:-1]
    new_batch.first_order_batch = torch.arange(len(graphs)).repeat_interleave((val_nnodes ** 0.5).long())
    return new_batch
