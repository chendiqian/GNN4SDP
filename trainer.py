import numpy as np
import torch
from torch_geometric.utils import unbatch
from torch_scatter import scatter


class PlainGNNTrainer:
    def __init__(self, accum):
        self.best_objgap = 1.e8
        self.patience = 0
        self.accum = accum

    def train(self, dataloader, model, optimizer, device):
        model.train()

        train_losses = 0.
        num_graphs = 0
        optimizer.zero_grad()
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_y, pred_S, M, N = model(data)
            loss = ((pred_S.reshape(-1) - data.dual_solution) ** 2).mean() \
                   + ((pred_y.reshape(-1) - data.y_solution) ** 2).mean()

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss = loss / self.accum
            loss.backward()
            if (i + 1) % self.accum == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

        return train_losses / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model, device):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        objgaps = 0.
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_y, pred_S, M, N = model(data)
            loss = ((pred_S.reshape(-1) - data.dual_solution) ** 2).mean() \
                   + ((pred_y.reshape(-1) - data.y_solution) ** 2).mean()

            val_losses += loss * data.num_graphs
            num_graphs += data.num_graphs

            # quick evaluation
            # the supervised y is -y
            obj_pred = -pred_y.sum(1) - M.sum((1, 2)) - N.sum((1, 2))
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()

        return val_losses / num_graphs, objgaps / num_graphs


class SSLDualTrainer:
    def __init__(self):
        self.best_objgap = 1.e8
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            y, S, M, N = model(data)
            loss = -y.sum(1) + M.sum((1, 2)) + N.sum((1, 2))

            train_losses += loss.detach().sum()
            num_graphs += data.num_graphs

            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model, device):
        model.eval()

        val_objs = 0.
        num_graphs = 0
        objgaps = 0.
        for i, data in enumerate(dataloader):
            data = data.to(device)

            y, S, M, N = model(data)
            obj_pred = y.sum(1) - M.sum((1, 2)) - N.sum((1, 2))
            num_graphs += data.num_graphs

            # quick evaluation
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()
            val_objs += obj_pred.sum()

        # print(torch.mean(M.sum((1, 2)) + N.sum((1, 2))).item())

        return val_objs / num_graphs, objgaps / num_graphs
