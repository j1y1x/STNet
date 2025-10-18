# pip install SciencePlots
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use(['science', 'ieee', 'grid'])
plt.style.use(['science', 'high-vis', 'grid'])

def plot_line(datas,
              data_labels,
              xy_labels,
              title,
              file_name,
              xlog=False,
              ylog=False):
    fig, ax = plt.subplots()
    if data_labels is not None:
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y, label=data_label)
        ax.legend()
    else:
        for data in datas:
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y)
    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()


def plot_density(datas,
              data_labels,
              xy_labels,
              title,
              file_name):
    fig, ax = plt.subplots()
    if data_labels is not None:
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            sns.kdeplot(x, label=data_label)
        ax.legend()
    else:
        for data in datas:
            x = data[:, 0]
            sns.kdeplot(x)
    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)

    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()
    
