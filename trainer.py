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

            pred_primal, pred_slack, pred_dual = model(data)
            loss = scatter((pred_primal - data.x_solution) ** 2, data.batch_dict['vals'], dim=0, reduce='mean').mean()
            if pred_slack is not None:
                loss = loss + scatter((pred_slack - data.dual_solution) ** 2, data.batch_dict['vals'], dim=0,
                                      reduce='mean').mean()
            if pred_dual is not None:
                loss = loss + scatter((pred_dual - data.y_solution) ** 2, data.batch_dict['cons'], dim=0,
                                      reduce='mean').mean()

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
    def eval(self, dataloader, model, device, project=False):
        model.eval()

        val_losses = 0.
        val_violations = 0.
        num_graphs = 0
        objgaps = 0.
        projected_objgaps = 0.
        projected_vios = 0.
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_primal, pred_slack, pred_dual = model(data)
            loss = scatter((pred_primal - data.x_solution) ** 2, data.batch_dict['vals'], dim=0, reduce='mean')
            if pred_slack is not None:
                loss = loss + scatter((pred_slack - data.dual_solution) ** 2, data.batch_dict['vals'], dim=0,
                                      reduce='mean')
            if pred_dual is not None:
                loss = loss + scatter((pred_dual - data.y_solution) ** 2, data.batch_dict['cons'], dim=0, reduce='mean')

            val_losses += loss.sum()
            num_graphs += data.num_graphs

            # quick evaluation
            obj_pred = scatter(pred_primal[data['obj', 'to', 'vals'].edge_index[1]] *
                               data['obj', 'to', 'vals'].edge_attr.squeeze(1),
                               data['obj', 'to', 'vals'].edge_index[0], dim=0, reduce='sum')
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()

            # violation
            Ax_minus_b = scatter(pred_primal[data['vals', 'to', 'cons'].edge_index[0]] *
                                 data['vals', 'to', 'cons'].edge_attr.squeeze(1),
                                 data['vals', 'to', 'cons'].edge_index[1], dim=0, reduce='sum',
                                 dim_size=data.b.shape[0]) - data.b
            val_violations += scatter(Ax_minus_b.abs(), data.batch_dict['cons'], dim=0, reduce='mean').sum()

            # project onto PSD cone
            if project:
                preds = unbatch(pred_primal, data.batch_dict['vals'], 0, data.num_graphs)

                x_projected = []
                for x in preds:
                    n2 = x.shape[0]
                    n = int(n2 ** 0.5)
                    x = x.cpu().numpy().reshape(n, n)
                    eigval, eigvec = np.linalg.eigh(x)
                    eigval = np.where(eigval < 0, 0., eigval)
                    x_p = (eigvec * eigval[None, :]) @ eigvec.T
                    x_projected.append(x_p.reshape(-1))
                x_projected = np.concatenate(x_projected, axis=0)
                x_projected = torch.from_numpy(x_projected).to(device).float()

                # quick evaluation
                obj_pred = scatter(x_projected[data['obj', 'to', 'vals'].edge_index[1]] *
                                   data['obj', 'to', 'vals'].edge_attr.squeeze(1),
                                   data['obj', 'to', 'vals'].edge_index[0], dim=0, reduce='sum')
                obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
                projected_objgaps += obj_gap.sum()

                # violation
                Ax_minus_b = scatter(x_projected[data['vals', 'to', 'cons'].edge_index[0]] *
                                     data['vals', 'to', 'cons'].edge_attr.squeeze(1),
                                     data['vals', 'to', 'cons'].edge_index[1], dim=0, reduce='sum',
                                     dim_size=data.b.shape[0]) - data.b
                projected_vios += scatter(Ax_minus_b.abs(), data.batch_dict['cons'], dim=0, reduce='mean').sum()

        if projected_objgaps:
            projected_objgaps = projected_objgaps / num_graphs
            projected_vios = projected_vios / num_graphs
        else:
            projected_objgaps = torch.tensor(0., device=device).float()
            projected_vios = torch.tensor(0., device=device).float()
        return val_losses / num_graphs, objgaps / num_graphs, projected_objgaps, val_violations / num_graphs, projected_vios


class SupervisedDualTrainer:
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

            pred_dual = model(data)
            loss = scatter((pred_dual - data.y_solution) ** 2, data.batch_dict['cons'], dim=0, reduce='mean').mean()

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

            pred_dual = model(data)
            loss = scatter((pred_dual - data.y_solution) ** 2, data.batch_dict['cons'], dim=0, reduce='mean')

            val_losses += loss.sum()
            num_graphs += data.num_graphs

            # quick evaluation
            obj_pred = -scatter(pred_dual * data.b, data.batch_dict['cons'], dim=0, reduce='sum')
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()

        return val_losses / num_graphs, objgaps / num_graphs


class SSLDualTrainer:
    def __init__(self, accum):
        self.best_objgap = 1.e8
        self.patience = 0
        self.accum = accum
        self.lamb = 10.

    def train(self, dataloader, model, optimizer, device):
        model.train()

        train_losses = 0.
        num_graphs = 0
        optimizer.zero_grad()
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_dual, eigvals = model(data)
            loss = -scatter(pred_dual * data.b, data.batch_dict['cons'], dim=0, reduce='sum').mean() \
                   + (torch.minimum(eigvals, torch.zeros_like(eigvals)) ** 2).sum(1).mean(0) * self.lamb

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

        val_objs = 0.
        num_graphs = 0
        objgaps = 0.
        psd_vio = 0.
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_dual, eigvals = model(data)
            num_graphs += data.num_graphs

            # quick evaluation
            obj_pred = -scatter(pred_dual * data.b, data.batch_dict['cons'], dim=0, reduce='sum')
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()
            val_objs = obj_pred.sum()

            psd_vio += (torch.minimum(eigvals, torch.zeros_like(eigvals)) ** 2).sum()

        return val_objs / num_graphs, objgaps / num_graphs, psd_vio / num_graphs
