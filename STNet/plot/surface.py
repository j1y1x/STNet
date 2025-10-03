# pip install SciencePlots
from matplotlib import cm
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee', 'grid'])
plt.style.use(['science', 'high-vis'])


def plot_surface_3D(X, Y, Z,
                    title,
                    xlabel,
                    ylabel,
                    zlabel,
                    file_name):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.9, cmap='rainbow')
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X,
    #                        Y,
    #                        Z,
    #                        rstride=1,
    #                        cstride=1,
    #                        cmap=cm.viridis,
    #                        linewidth=0,
    #                        antialiased=False)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    fig.savefig(file_name + '.png', dpi=300)
