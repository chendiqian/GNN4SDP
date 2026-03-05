import torch


class SSLDualTrainer:
    def __init__(self):
        self.best_obj = -1.e8
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_X, pred_y, pred_S, C, b = model(data)
            loss = -(pred_y * b).sum(1)

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

            pred_X, pred_y, pred_S, C, b = model(data)
            obj_pred = (pred_y * b).sum(1)
            num_graphs += data.num_graphs

            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()
            val_objs += obj_pred.sum()

        return val_objs / num_graphs, objgaps / num_graphs

    def step(self, val_obj):
        # maximizing the dual obj
        if self.best_obj < val_obj:
            self.patience = 0
            self.best_obj = val_obj
        else:
            self.patience += 1


class SSLPrimalTrainer:
    def __init__(self):
        self.best_obj = 1.e8
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_X, pred_y, pred_S, C, b = model(data)
            # the padding of C are 0s, so it is find if pred_X has some nonzero paddings
            loss = (pred_X * C).sum((1, 2)).mean()
            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            optimizer.zero_grad()
            loss.backward()
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

            pred_X, pred_y, pred_S, C, b = model(data)
            num_graphs += data.num_graphs

            # quick evaluation
            obj_pred = (pred_X * C).sum((1, 2))
            obj_gt = data.obj_solution
            obj_gap = (obj_pred - obj_gt).abs() / obj_gt.abs()
            objgaps += obj_gap.sum()
            val_objs += obj_pred.sum()

        return val_objs / num_graphs, objgaps / num_graphs

    def step(self, val_obj):
        # minimizing the dual obj
        if self.best_obj > val_obj:
            self.patience = 0
            self.best_obj = val_obj
        else:
            self.patience += 1


class SSLPrimalDualTrainer(SSLDualTrainer):
    # can evaluate on either primal or dual, we choose dual
    def train(self, dataloader, model, optimizer, device):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred_X, pred_y, pred_S, C, b = model(data)

            loss = (pred_X * pred_S).sum((1, 2))

            train_losses += loss.detach().sum()
            num_graphs += data.num_graphs

            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses / num_graphs
