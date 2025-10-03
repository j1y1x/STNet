from cmath import inf
import logging
import sys
import time
import numpy as np
from scipy.interpolate import griddata
import torch
from plot.line import plot_density, plot_line
from plot.heatmap import plot_heatmap3
from model_config import PINNConfig

from init_config import TASK_NAME, get_device, path, root_path
from train_config import *
from pyDOE import lhs

def log(obj):
    print(obj)
    logging.info(obj)

if __name__ == "__main__":
    start_time = time.time()
    device = get_device(sys.argv)

    # TIME_STR = '20220913_082821'
    # TIME_STR = '20220913_082829'
    # TIME_STR = '20220913_082835'
    TIME_STR = '20220913_082842'
    path = '/' + TASK_NAME + '/' + TIME_STR + '/'
        
    if d==1:
        GRID_SIZE = 1000
        step = 1
        x = np.linspace(lb[0], ub[0], GRID_SIZE)
        x = x.reshape(x.shape[0], 1)
    elif d==2:
        GRID_SIZE = 400
        X = np.linspace(lb[0], ub[0], GRID_SIZE)
        Y = np.linspace(lb[1], ub[1], GRID_SIZE)
        X_VALID, Y_VALID = np.meshgrid(X, Y)
        X_valid = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
        x = X_valid
    else:
        # N_R = 200000
        N_R = 100000
        # N_R = 20000
        x = lb + (ub-lb)*lhs(d, N_R)
    x_star = x
    X = []
    for i in range(d):
        xi = x[:, i:i+1]
        X.append(xi)

    u_true = Exact_u(X)
    u_true = u_true/np.linalg.norm(u_true, ord=2)

    net_path = root_path + '/' + path + '/PINN.pkl'
    model_config = PINNConfig.reload_config(net_path=net_path)

    X1 = []
    for i in range(d):
        xi = x[:, i:i+1]
        X1.append(model_config.data_loader(xi))
    u_pred = model_config.forward(X1)

    u_pred = torch.sign(torch.mean(u_pred)) * u_pred/torch.norm(u_pred, p=2)
    # t = torch.sum(u_pred**2)
    u_pred = model_config.detach(u_pred)

    # u_pred = u_pred * (u_pred.shape[0]/np.sum(u_pred))
    # u_true = u_true * (u_true.shape[0]/np.sum(u_true))
    u_pred = u_pred * np.sqrt(u_pred.shape[0])
    u_true = u_true * np.sqrt(u_true.shape[0])

    L_infinity_u = np.linalg.norm(u_true-u_pred, ord=inf)
    L_2_u = np.sqrt(np.linalg.norm(u_true-u_pred, ord=2)**2/u_pred.shape[0])
    
    L_infinity_lambda = np.abs(lambda_-model_config.lambda_)
    L_2_lambda = (lambda_-model_config.lambda_)**2
    log_str = 'lambda ' + str(model_config.lambda_) +\
        ' L_infinity_lambda '+str(L_infinity_lambda) +' L_2_lambda '+str(L_2_lambda) +\
        ' L_infinity_u ' + str(L_infinity_u) + ' L_2_u ' + str(L_2_u)
    log(log_str)


    if d==1:
        file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/pm_line'
        datas = []
        data_labels = ['u_true', 'u_pred']
        data = np.stack((x_star, u_true), 1)
        datas.append(data)
        data = np.stack((x_star, u_pred), 1)
        datas.append(data)
    
        xy_labels = ['x', 'u']
        plot_line(datas=datas,
                data_labels=data_labels,
                xy_labels=xy_labels,
                title=None,
                file_name=file_name,
                ylog=False)
    if d==2:
        U_star = griddata(x_star, u_true.flatten(), (X_VALID, Y_VALID), method='cubic')
        U_pred = griddata(x_star, u_pred.flatten(), (X_VALID, Y_VALID), method='cubic')
        file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/pm_heatmap3'
        # plot_heatmap3(X, Y, T, P, E=None, xlabel=None, ylabel=None, T_title=None, P_title=None, E_title=None, file_name=None, abs=True):
        plot_heatmap3(X=X_VALID, Y=Y_VALID, T=U_star, P=U_pred, E=None, xlabel='x',
                    ylabel='y', file_name=file_name)


    u_list = [u_true, u_pred]
    data_labels = ['u_true', 'u_pred']
    min_u = np.min(u_list)
    max_u = np.max(u_list)
    N = 100
    x = np.linspace(min_u, max_u, N+1)
    delta_x = (max_u-min_u)/N
    density_list = []
    datas = []
    for u in u_list:
        density = np.zeros(N+1)
        for i in range(u.shape[0]):
            value = u[i, 0]
            j = round((value-min_u)/delta_x)
            density[j] = density[j] + 1
        density_list.append(density)
    max_d = np.max(density_list)
    for density in density_list:
        data = np.stack((x, density/max_d), 1)
        datas.append(data)
    xy_labels = ['u', 'density']
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/density'
    plot_line(datas=datas,
            data_labels=data_labels,
            xy_labels=xy_labels,
            title=None,
            file_name=file_name,
            ylog=False)


    import scipy.stats as stats
    y=stats.norm.pdf(u_true)
    file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/pm_density_' + str(d)
    plot_density(datas=u_list,
            data_labels=data_labels,
            xy_labels=xy_labels,
            title=None,
            file_name=file_name)


    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
