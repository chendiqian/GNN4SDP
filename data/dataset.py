import os
import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Batch, InMemoryDataset


class LPDataset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            split: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'valid.pt', 'test.pt']

    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.processed_dir) if n.startswith('batch')])
        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))

        lens = len(data_list)
        torch.save(self.collate(data_list[:int(0.8 * lens)]), osp.join(self.processed_dir, 'train.pt'))
        torch.save(self.collate(data_list[int(0.8 * lens): int(0.9 * lens)]), osp.join(self.processed_dir, 'valid.pt'))
        torch.save(self.collate(data_list[int(0.9 * lens):]), osp.join(self.processed_dir, 'test.pt'))
