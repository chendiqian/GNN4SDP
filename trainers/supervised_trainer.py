import torch
from torch_scatter import scatter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        for i, data in enumerate(dataloader):
            data = data.to(device)
            pred = model(data)
            loss = scatter((pred - data.x_solution) ** 2, data.batch_dict['vals'], dim=0, reduce='mean')
            val_losses += loss.sum()
            num_graphs += data.num_graphs

            batched_c = pred.new_zeros(*pred.shape)
            batched_c[data['obj', 'to', 'vals'].edge_index[1]] = data['obj', 'to', 'vals'].edge_attr.squeeze(1)
            obj_pred = scatter(pred * batched_c, data.batch_dict['vals'], dim=0, reduce='sum')
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / torch.maximum(obj_gt, obj_pred).abs()
            objgaps.append(obj_gap)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        return val_losses.item() / num_graphs, objgaps
