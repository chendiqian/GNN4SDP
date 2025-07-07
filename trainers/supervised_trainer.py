from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import cumsum, degree, unbatch
from torch_scatter import scatter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def unbatch_edge_index(
        edge_index: Tensor,
        edge_attr: Tensor,
        src_batch: Tensor,
        dst_batch: Tensor,
        batch_size: Optional[int] = None,
):
    deg = degree(dst_batch, batch_size, dtype=torch.long)
    ptr = cumsum(deg)

    edge_batch = src_batch[edge_index[0]]
    edge_index = edge_index[1] - ptr[edge_batch]
    sizes = degree(edge_batch, batch_size, dtype=torch.long).cpu().tolist()
    return edge_index.split(sizes, dim=0), edge_attr.split(sizes, dim=0)


class PlainGNNTrainer:
    def __init__(self):
        self.best_objgap = 1.e8
        self.patience = 0

    def train_step(self, data, label, model, optimizer):
        optimizer.zero_grad()
        pred = model(data)
        loss = scatter((pred - label) ** 2, data.batch_dict['vals'], dim=0, reduce='mean')
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            label = data.x_solution
            loss = self.train_step(data, label, model, optimizer)
            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs

        return train_losses.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        objgaps = []
        projected_objgaps = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            pred = model(data)
            loss = scatter((pred - data.x_solution) ** 2, data.batch_dict['vals'], dim=0, reduce='mean')
            val_losses += loss.sum()
            num_graphs += data.num_graphs

            obj_pred = scatter(pred[data['obj', 'to', 'vals'].edge_index[1]] *
                               data['obj', 'to', 'vals'].edge_attr.squeeze(1),
                               data['obj', 'to', 'vals'].edge_index[0], dim=0, reduce='sum')

            preds = unbatch(pred, data.batch_dict['vals'], 0, data.num_graphs)
            edges, coeffs = unbatch_edge_index(data['obj', 'to', 'vals'].edge_index,
                                               data['obj', 'to', 'vals'].edge_attr.squeeze(1),
                                               data.batch_dict['obj'],
                                               data.batch_dict['vals'],
                                               data.num_graphs)

            objs = []
            for x, edge, coeff in zip(preds, edges, coeffs):
                n2 = x.shape[0]
                n = int(n2 ** 0.5)
                x = x.cpu().numpy().reshape(n, n)
                edge = edge.cpu().numpy()
                eigval, eigvec = np.linalg.eigh(x)
                mask = eigval >= 0.
                eigval = eigval[None, mask]
                eigvec = eigvec[:, mask]
                src = edge // n
                dst = edge % n
                x_projected = (eigvec[src] * eigval * eigvec[dst]).sum(1)
                obj = (x_projected * coeff.cpu().numpy()).sum()
                objs.append(obj)

            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / torch.maximum(obj_gt.abs(), obj_gt.abs())
            obj_gt = obj_gt.cpu().numpy()
            objs = np.array(objs, dtype=np.float32)
            projected_objgaps.append(np.abs(objs - obj_gt) / np.maximum(np.abs(objs), np.abs(obj_gt)))
            objgaps.append(obj_gap)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        projected_objgaps = np.concatenate(projected_objgaps, axis=0).mean().item()
        return val_losses.item() / num_graphs, objgaps, projected_objgaps
