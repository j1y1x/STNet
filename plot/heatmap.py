# pip install SciencePlots
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee', 'grid'])
plt.style.use(['science', 'high-vis', 'grid'])

'''
X: 二维矩阵
Y: 二维矩阵
Z: 数据
scatter: 撒点坐标
'''


def plot_heatmap(X, Y, Z, xlabel=None, ylabel=None, title=None, file_name=None, scatter_x=None, scatter_y=None):
    fig, ax = plt.subplots()
    cset = plt.contourf(X, Y, Z)
    plt.colorbar(cset)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
        
    if scatter_x is not None and len(scatter_x) != 0:
        ax.scatter(scatter_x, scatter_y)

    if file_name is None:
        plt.show()
    else:
        fig.savefig(file_name + '.png', dpi=300)


'''
X: 二维矩阵
Y: 二维矩阵
T: 真实值
P: 预测值
E: 误差，可以不输入
abs: abs(E)
'''


def plot_heatmap3(X, Y, T, P, E=None, xlabel=None, ylabel=None, T_title=None, P_title=None, E_title=None, file_name=None, abs=True):
    fig = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    cset = plt.contourf(X, Y, T)
    plt.colorbar(cset)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if T_title is None:
        T_title = 'True'
    plt.title(T_title)

    plt.subplot(1, 3, 2)
    cset = plt.contourf(X, Y, P)
    plt.colorbar(cset)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if P_title is None:
        P_title = 'Pred'
    plt.title(P_title)

    if E is None:
        E = T - P

    if abs:
        E = np.abs(E)

    plt.subplot(1, 3, 3)
    cset = plt.contourf(X, Y, E)
    plt.colorbar(cset)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if E_title is None:
        E_title = 'Error'
    plt.title(E_title)

    plt.tight_layout()

    if file_name is None:
        plt.show()
    else:
        fig.savefig(file_name + '.png', dpi=300)
