import os
import torch
import numpy as np
from torch import autograd
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)


class BaseConfig:
    def __init__(self):
        super().__init__()
        self.loss = None
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None
        self.params = None

    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float64)
        return x_tensor.to(self.device)

    def coor_shift(self, X, lb, ub):
        X_shift = 2.0 * (X - lb) / (ub - lb) - 1.0
        # X_shift = torch.from_numpy(X_shift).float().requires_grad_()
        return X_shift

    def detach(self, data):
        tmp_data = data.detach().cpu().numpy()
        if np.isnan(tmp_data).any():
            raise Exception
        return tmp_data

    def loss_func(self, pred_, true_=None):
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
            # true_ = self.data_loader(true_)
        return self.loss_fn(pred_, true_)

    def compute_grad(self, u, x):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x

    def optimize_one_epoch(self):
        return self.loss

    def train_Adam(self, params, Adam_steps = 50000, Adam_init_lr = 1e-3, scheduler_name=None, scheduler_params=None):
        Adam_optimizer = torch.optim.Adam(params=params,
                                        lr=Adam_init_lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                      amsgrad=False)
        self.optimizer = Adam_optimizer
        self.optimizer_name = 'Adam'
        if scheduler_name == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            Adam_scheduler = MultiStepLR(Adam_optimizer, **scheduler_params)
        else:
            Adam_scheduler = None
        self.scheduler = Adam_scheduler
        for it in range(Adam_steps):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def train_LBFGS(self, params, LBFGS_steps = 10000, LBFGS_init_lr = 1, tolerance_LBFGS = -1, LBFGS_scheduler=None):
        LBFGS_optimizer = torch.optim.LBFGS(
            params=params,
            lr=LBFGS_init_lr,
            max_iter=LBFGS_steps,
            tolerance_grad=tolerance_LBFGS,
            tolerance_change=tolerance_LBFGS,
            history_size=100,
            line_search_fn=None)
        self.optimizer = LBFGS_optimizer

        self.optimizer_name = 'LBFGS'
        self.scheduler = LBFGS_scheduler

        def closure():
            loss = self.optimize_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss
        try:
            self.optimizer.step(closure)
        except Exception as e:
            print(e)

    def save(net, path, name='PINN',vec=None,id=None):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(net, path + '/' + name + '.pkl')
        if id==1:
            torch.save(vec, path + '/' + name + '_vec1.pkl')
        elif id==2:
            torch.save(vec, path + '/' + name + '_vec2.pkl')
        elif id==3:
            torch.save(vec, path + '/' + name + '_vec3.pkl')
        elif id==4:
            torch.save(vec, path + '/' + name + '_vec4.pkl')


    @staticmethod
    def reload_config(net_path):
        net = torch.load(net_path)
        return net
        # state_dict = torch.load(net_path)
        # model.load_state_dict(state_dict)
