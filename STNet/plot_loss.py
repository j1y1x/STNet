import numpy as np
from plot.line import plot_line

from init_config import TASK_NAME, path, root_path
from train_config import *


def read_PINN_log(PINN_log_path, step=1):
    keys = ['Loss']
    values = []
    with open(PINN_log_path, 'r') as fs:
        line_num = 0
        while True:
            line = fs.readline()
            if not line:
                break
            if 'Iter ' in line:
                line_num = line_num + 1
                if line_num % step != 0:
                    continue
                datas = line.split(' Loss ')[-1].split(' ')
                if datas is not None:
                    if len(keys) == 1:
                        keys[1:] = datas[1::2]
                    values.append(datas[0::2])
    values = np.array(values, dtype=float)
    keys = np.array(keys)
    return keys, values



if __name__ == "__main__":
    datas = []
    TIME_STR = '20220913_082821'
    path = '/' + TASK_NAME + '/' + TIME_STR + '/'

    PINN_log_path = root_path + '/' + path + '/log.txt'

    keys, values = read_PINN_log(
        PINN_log_path=PINN_log_path, step=1)

    Num_Epoch = len(values)
    epochs = [i for i in range(1, Num_Epoch + 1, 1)]
    epochs = np.array(epochs)

# INFO:root:Adam Iter 50000 Loss 2.4380576e-12 lambda_ 9.869534 loss_IPM 2.4380576e-12 min_loss 2.3794005e-12 lambda 9.869529 abs_lambda 7.563064258064855e-05 rel_lambda 7.662986225901904e-06 LR 0.001
    # idx = [3, 4, 5, 6, 7, 9, 10]
    idx = [0]
    t_keys = keys[idx]
    # t_values = values[:, idx]
    step = 50
    for i in idx:
        value = values[:, i]
        data = np.stack((epochs, value), 1)
        data[:, 0:1] = data[:, 0:1] * 10
        data = data[:4400, :]
        datas.append(data[::step, :])

    data_labels = t_keys[:]
    # data_labels = None

    xy_labels = ['Epoch', 'Loss']
    plot_line(datas=datas,
              data_labels=data_labels,
              xy_labels=xy_labels,
              title=None,
              file_name=root_path + '/' + path + '/loss',
              ylog=True)
    print('done')
