import logging
import sys
import argparse
import numpy as np
from pyDOE import lhs
from init_config import get_device, path, root_path, init_log, train_Adam

def log(obj):
    print(obj)
    logging.info(obj)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train STNet model')
    parser.add_argument('--d', type=int, required=True, help='dimension')
    parser.add_argument('--problem_type', type=str, default='Harmonic', help='problem_type')
    parser.add_argument('--device', type=str, default=None, help='device')
    args = parser.parse_args()
    d = args.d
    problem_type = args.problem_type
    user_device = args.device


    init_log()
    if user_device is not None:
        device = user_device if user_device in ['cpu', 'cuda'] else 'cpu'
    else:
        device = get_device(sys.argv)
    if problem_type == 'Harmonic':
        lb = np.array([0 for i in range(d)])
        ub = np.array([1 for i in range(d)])
    elif problem_type == 'Oscillator':
        lb = np.array([-5 for i in range(d)])
        ub = np.array([5 for i in range(d)])
    elif problem_type == 'Planck':
        lb = np.array([0 for i in range(d)])
        ub = np.array([2*np.pi for i in range(d)])
        k = 3
    param_dict = {
        'lb': lb,
        'ub': ub,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    if d==1:
        N_R = 20000
    elif d==2:
        N_R = 40000
    elif d==5:
        N_R = 59049
    else:
        N_R = 59049
    x = lb + (ub-lb)*lhs(d, N_R)

    log_str = f' d {d} problem_type {problem_type} N_R {N_R}'
    log(log_str)

    train_dict = {
        'x': x,
        'd': d,
        'N_R': N_R,
    }

    if d==1 or d==2:
        if problem_type == 'Harmonic' or problem_type == 'Oscillator':
            layers = [d, 20, 20, 20, 20, 1]
        elif problem_type == 'Planck':
            layers = [2*d*k, 20, 20, 20, 20, 1]
    elif d==5:
        if problem_type == 'Harmonic' or problem_type == 'Oscillator':
            layers = [d, 40, 40, 40, 40, 1]
        elif problem_type == 'Planck':
            layers = [2*d*k, 15, 15, 15, 15, 15, 1]
    else:
        layers = [d, 80, 80, 80, 80, 1]

    log(layers)

    if d==1:
        train_Adam(layers, device, param_dict, train_dict, problem_type, Adam_steps=400000, Adam_init_lr=1e-3)
    elif d==2:
        train_Adam(layers, device, param_dict, train_dict, problem_type, Adam_steps=300000, Adam_init_lr=1e-3)
    elif d==5:
        train_Adam(layers, device, param_dict, train_dict, problem_type, Adam_steps=400000, Adam_init_lr=1e-3)
    else:
        train_Adam(layers, device, param_dict, train_dict, problem_type, Adam_steps=500000, Adam_init_lr=1e-3)