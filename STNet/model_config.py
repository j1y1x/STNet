import logging
import time
import numpy as np
import torch
from base_config import BaseConfig



def log(obj):
    print(obj)
    logging.info(obj)


class PINNConfig(BaseConfig):
    def __init__(self, param_dict, train_dict, model, model_id):
        super().__init__()
        self.init(loss_name='sum')
        self.model = model
        self.model_id = model_id
        lb, ub, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        x, self.d, self.N_R = self.unzip_train_dict(train_dict=train_dict)
        self.x_squared_sum = self.data_loader(np.sum(x ** 2, axis=1).reshape(self.N_R, 1), requires_grad=False)
        self.x = []
        for i in range(self.d):
            xi = x[:, i:i+1]
            X = self.data_loader(xi)
            self.x.append(X)

        u = np.random.rand(self.N_R, 1)
        u = u/np.linalg.norm(u)
        self.u = self.data_loader(u, requires_grad=False)
        self.lambda_last = 1
        self.shift1 = 0
        self.shift2 = 0
        self.shift3 = 0
        self.shift4 = 0
        self.tmp_lambda = 0
        self.lambda_ = None
        self.eigenvec = None


    def init(self, loss_name='mean', model_name='PINN'):
        self.start_time = None
        self.min_loss = 1e20
        self.nIter = 0
        if loss_name == 'sum':
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        else:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.model_name = model_name



    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['lb'], param_dict['ub'],
                      param_dict['device'], param_dict['path'],
                      param_dict['root_path'])
        return param_data

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x'],
            train_dict['d'],
            train_dict['N_R'],
        )
        return train_data

    def net_model(self, x, problem_type=None):
        if problem_type == 'Harmonic' or problem_type == 'Oscillator':
            if isinstance(x, list):
                X = torch.cat((x), 1)
            else:
                X = x
            X = self.coor_shift(X, self.lb, self.ub)
            result = self.model.forward(X)
            g_x = 1
            for i in range(self.d):
                # g_x = g_x * (1-torch.exp(-(x[i]-self.lb[i])))*(1-torch.exp(-(x[i]-self.ub[i])))
                g_x = g_x * (torch.exp((x[i] - self.lb[i])) - 1) * (torch.exp(-(x[i] - self.ub[i])) - 1)
            result = g_x * result
        elif problem_type == 'Planck':
            if isinstance(x, list):
                X = torch.cat((x), 1)
            else:
                X = x
            lb = self.lb
            ub = self.ub
            for i in range(5):
                lb = torch.cat([self.lb, lb], dim=0)
                ub = torch.cat([self.ub, ub], dim=0)
            X = self.coor_shift(X, lb, ub)
            result = self.model.forward(X)
        return result

    def forward(self, x, problem_type=None):
        result = self.net_model(x, problem_type)
        return result

    def optimize_one_epoch(self,problem_type, max_val=None, max_vec=None, max_val1=None, max_vec1=None,max_val2=None,max_vec2=None,optimizer=None):
        if self.start_time is None:
            self.start_time = time.time()

        self.optimizer = optimizer
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()
        if problem_type == 'Harmonic':
            x = self.x
            u = self.forward(x, problem_type)
            u_xx = None
            for i in range(self.d):
                xi = x[i]
                u_xi = self.compute_grad(u, xi)
                u_xixi = self.compute_grad(u_xi, xi)
                if u_xx is None:
                    u_xx = u_xixi
                else:
                    u_xx = u_xx + u_xixi
        elif problem_type == 'Oscillator':
            x = self.x
            u = self.forward(x, problem_type)
            u_xx = None
            for i in range(self.d):
                xi = x[i]
                u_xi = self.compute_grad(u, xi)
                u_xixi = self.compute_grad(u_xi, xi)
                if u_xx is None:
                    u_xx = u_xixi
                else:
                    u_xx = u_xx + u_xixi
                xu = self.x_squared_sum * u
                u_xx = (u_xx - xu) / 2.0
        elif problem_type == 'Planck':
            x = self.x
            x_net = torch.stack(x, dim=1)
            cos_x = torch.cos(x_net).to(self.device)
            cos2_x = torch.cos(2 * x_net).to(self.device)
            cos3_x = torch.cos(3 * x_net).to(self.device)
            sin_x = torch.sin(x_net).to(self.device)
            sin2_x = torch.sin(2 * x_net).to(self.device)
            sin3_x = torch.sin(3 * x_net).to(self.device)
            x_net1 = torch.cat([cos_x, cos2_x, cos3_x, sin_x, sin2_x, sin3_x], dim=1).squeeze(-1)
            u = self.forward(x_net1, problem_type)
            u_xx = None
            ker = None
            uv = None
            v_xx = None
            for i in range(self.d):
                if ker is None:
                    ker = torch.cos(x[i])
                else:
                    ker = ker + torch.cos(x[i])
            v = torch.sin(ker)
            for i in range(self.d):
                xi = x[i]
                u_xi = self.compute_grad(u, xi)
                u_xixi = self.compute_grad(u_xi, xi)
                if u_xx is None:
                    u_xx = u_xixi
                else:
                    u_xx = u_xx + u_xixi
                # v_xi = self.compute_grad(v,xi)
                v_xi = -torch.cos(ker) * torch.sin(xi)
                if uv is None:
                    uv = u_xi * v_xi
                else:
                    uv = uv + u_xi * v_xi
                v_xixi = self.compute_grad(v_xi, xi)
                if v_xx is None:
                    v_xx = v_xixi
                else:
                    v_xx = v_xx + v_xixi
                u_xx = u_xx + uv + v_xx*u
        if self.model_id == 1:
            Lu = -u_xx - self.shift1 * u
        elif self.model_id == 2:
            Lu = -u_xx - max_val*torch.matmul(max_vec,
                                      torch.matmul(max_vec.transpose(0, 1), u)) - self.shift2 * u
        elif self.model_id == 3:
            Lu = -u_xx - max_val*torch.matmul(max_vec,
                                      torch.matmul(max_vec.transpose(0, 1), u)) - max_val1*torch.matmul(
                 max_vec1, torch.matmul(max_vec1.transpose(0, 1), u)) - self.shift3 * u
        elif self.model_id == 4:
            Lu = -u_xx - max_val*torch.matmul(max_vec,
                                      torch.matmul(max_vec.transpose(0, 1), u)) - max_val1*torch.matmul(
                max_vec1,
                torch.matmul(max_vec1.transpose(0, 1), u)) - max_val2*torch.matmul(
                max_vec2, torch.matmul(max_vec2.transpose(0, 1), u))
        tmp_loss = self.loss_func(Lu - self.lambda_last * self.u)
        u1 = Lu
        u1 = u1 / torch.norm(u1, p=2)

        loss_PM = self.loss_func(u1, self.u)
        self.u = self.data_loader(self.detach(u), requires_grad=False)
        alpha_PM = 1
        self.loss = loss_PM * alpha_PM

        self.loss.backward()
        self.nIter = self.nIter + 1
        Luu = torch.sum(Lu * self.u)
        uu = torch.sum(u ** 2)
        lambda_ = self.detach(Luu / uu)
        lambda_ = lambda_.max()
        self.lambda_last = lambda_

        if self.nIter % (self.N_R / 20) == 0:
            if self.model_id == 1:
                self.shift1 = lambda_ - 2
            elif self.model_id == 2:
                self.shift2 = lambda_ - 2
            elif self.model_id == 3:
                self.shift3 = lambda_ - 2
            self.lambda_last = 1
        if self.model_id == 1:
            self.tmp_lambda = lambda_ + self.shift1
        elif self.model_id == 2:
            self.tmp_lambda = lambda_ + self.shift2
        elif self.model_id == 3:
            self.tmp_lambda = lambda_ + self.shift3
        elif self.model_id == 4:
            self.tmp_lambda = lambda_ + self.shift4
        loss = self.detach(tmp_loss)
        if self.nIter == 1:
            self.lambda_ = lambda_
            self.eigenvec = self.u
        if loss < self.min_loss:
            if self.nIter > self.N_R / 20:
                if self.model_id == 1:
                    self.lambda_ = lambda_ + self.shift1
                elif self.model_id == 2:
                    self.lambda_ = lambda_ + self.shift2
                elif self.model_id == 3:
                    self.lambda_ = lambda_ + self.shift3
            else:
                self.lambda_ = lambda_
            self.min_loss = loss
            self.eigenvec = self.u
            PINNConfig.save(net=self,
                            path=self.root_path + '/' + self.path,
                            name=self.model_name,
                            vec=self.eigenvec,
                            id=self.model_id)

        loss_remainder = 10
        if np.remainder(self.nIter, loss_remainder) == 0:
            loss_PM = self.detach(loss_PM)
            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) + \
                      ' tmp_lambda ' + str(self.tmp_lambda) + ' loss_PM ' + str(loss_PM) + \
                      ' min_loss ' + str(self.min_loss) + \
                      ' lambda ' + str(self.lambda_) + \
                      ' LR ' + str(self.optimizer.state_dict()['param_groups'][0]['lr'])

            log(log_str)

            elapsed = time.time() - self.start_time
            print('Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder))
            logging.info('Time: %.4f s Per %d Iterators' %
                         (elapsed, loss_remainder))
            self.start_time = time.time()
        return self.loss,self.lambda_,self.eigenvec
