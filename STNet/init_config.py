import os
import os.path
import datetime
import logging
import os
import random
import time
import numpy as np
import torch
from model import MLP, ResNet
from model_config import PINNConfig



def getCurDirName():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # curDirName = curDir.split(parentDir)[-1]
    curDirName = os.path.split(curDir)[-1]
    return curDirName




def getParentDir():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # this will return parent directory.
    parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
    # parentDirName = os.path.split(parentDir)[-1]
    return parentDir




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



setup_seed(0)
TASK_NAME = 'task_' + getCurDirName()
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = getParentDir() + '/data/'
path = '/' + TASK_NAME + '/' + now_str + '/'
log_path = root_path + '/' + path



def init_log():
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log.txt'),
                        level=logging.INFO)



def get_device(argv):
    if len(argv) > 1 and 'cuda' == argv[1] and torch.cuda.is_available(
    ):
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    logging.info('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')
    return device




def train_Adam(layers, device, param_dict, train_dict, problem_type, Adam_steps=50000,  Adam_init_lr=1e-3, scheduler_name=None, scheduler_params=None):
    start_time = time.time()
    model1 = MLP(layers)
    model1.to(device)
    model2 = MLP(layers)
    model2.to(device)
    model3 = MLP(layers)
    model3.to(device)
    model4 = MLP(layers)
    model4.to(device)
    model_config1 = PINNConfig(param_dict=param_dict,
                               train_dict=train_dict, model=model1, model_id=1)
    model_config2 = PINNConfig(param_dict=param_dict,
                               train_dict=train_dict, model=model2, model_id=2)
    model_config3 = PINNConfig(param_dict=param_dict,
                               train_dict=train_dict, model=model3, model_id=3)
    model_config4 = PINNConfig(param_dict=param_dict,
                               train_dict=train_dict, model=model4, model_id=4)

    if model_config1.params is not None:
        params1 = model_config1.params
        params2 = model_config2.params
        params3 = model_config3.params
        params4 = model_config4.params
    else:
        params1 = model1.parameters()
        params2 = model2.parameters()
        params3 = model3.parameters()
        params4 = model4.parameters()
        optimizer1 = torch.optim.Adam(params=params1, lr=Adam_init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                      amsgrad=False,foreach=False)
        optimizer2 = torch.optim.Adam(params=params2, lr=Adam_init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                      amsgrad=False,foreach=False)
        optimizer3 = torch.optim.Adam(params=params3, lr=Adam_init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                      amsgrad=False,foreach=False)
        optimizer4 = torch.optim.Adam(params=params4, lr=Adam_init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                      amsgrad=False,foreach=False)


        for it in range(Adam_steps):
            optimizer1.zero_grad()
            loss1, max_val, max_vec = model_config1.optimize_one_epoch(problem_type, optimizer=optimizer1)
            optimizer1.step()

            optimizer2.zero_grad()
            loss2, max_val1, max_vec1 = model_config2.optimize_one_epoch(problem_type, max_val, max_vec, optimizer=optimizer2)
            optimizer2.step()

            optimizer3.zero_grad()
            loss3, max_val2, max_vec2 = model_config3.optimize_one_epoch(problem_type, max_val, max_vec, max_val1, max_vec1, optimizer=optimizer3)
            optimizer3.step()

            optimizer4.zero_grad()
            loss4, max_val3, max_vec3 = model_config4.optimize_one_epoch(problem_type, max_val, max_vec, max_val1, max_vec1, max_val2, max_vec2, optimizer=optimizer4)
            optimizer4.step()

    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))

