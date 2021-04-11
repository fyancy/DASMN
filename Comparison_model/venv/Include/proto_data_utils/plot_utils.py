from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def set_figure(font_size=10., tick_size=8., ms=7., lw=1.2, fig_w=8.):
    # lw: linewidth, 1., 1.2
    # print(plt.rcParams.keys())  # 很有用，查看所需属性
    # exit()
    cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
    w = fig_w * cm_to_inc  # cm ==> inch
    h = w * 3 / 4
    plt.rcParams['figure.figsize'] = (w, h)  # 单位 inc
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['figure.figsize'] = (14 * cm_to_inc, 6 * cm_to_inc)

    plt.rc('font', family='Times New Roman', weight='normal', size=str(font_size))
    plt.rcParams['axes.linewidth'] = lw  # 图框宽度

    # plt.rcParams['lines.markeredgecolor'] = 'k'
    plt.rcParams['lines.markeredgewidth'] = lw
    plt.rcParams['lines.markersize'] = ms

    # 刻度在内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['xtick.major.size'] = 2.5  # 刻度长度

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['ytick.major.size'] = 2.5

    plt.rcParams["legend.frameon"] = True  # 图框
    plt.rcParams["legend.framealpha"] = 0.8  # 不透明度
    plt.rcParams["legend.fancybox"] = False  # 圆形边缘
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams["legend.columnspacing"] = 1  # /font unit 以字体大小为单位
    plt.rcParams['legend.labelspacing'] = 0.2
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.borderpad"] = 0.3


# for domain adaptation
def tSNE_fun(input_data, shot, name=None, labels=None, n_dim=2):
    """
    :param shot:
    :param labels:
    :param input_data:  (n, dim)
    :param name: name
    :param n_dim: 2d or 3d
    :return: figure
    """
    t0 = time()
    classes = input_data.shape[0] // shot
    # da = umap.UMAP(n_neighbors=shot, n_components=n_dim, random_state=0).fit_transform(input_data)
    da = TSNE(n_components=n_dim, perplexity=shot, init='pca', random_state=0,
              angle=0.3).fit_transform(input_data)  # (n, n_dim)
    da = MinMaxScaler().fit_transform(da)  # [0, 1]

    color_set = [
        [0.00, 0.45, 0.74],  # 蓝色
        [0.93, 0.69, 0.13],  # 黄色
        [0.85, 0.33, 0.10],  # 橘红色
        [0.49, 0.18, 0.56],  # 紫色
        [0.47, 0.67, 0.19],  # 绿色
        [0.30, 0.75, 0.93],  # 淡蓝色
        [0.64, 0.08, 0.18],  # 棕色
    ]
    color = [
        [0.00, 0.45, 0.74],  # 蓝色
        [0.64, 0.08, 0.18],  # 棕色
        [0.46, 0.65, 0.20],  # 绿色
        [0.30, 0.75, 0.93],  # 淡蓝色
        [0.85, 0.33, 0.10],  # 橘红色
        [0.73, 0.92, 0.47],  # 淡绿色
    ]
    # color = color_set[:classes // 2] + color_set[:classes // 2]
    color = np.asarray(color)
    color = np.tile(color[:classes][:, None], (1, shot, 1)).reshape(-1, 3)
    mark = ['o', '^', '.', 'v', 's', 'D']
    # method 1:
    # m1 = [mark[0]] * (classes // 2)
    # m2 = [mark[1]] * (classes // 2)
    # mark = m1 + m2
    # method 2:
    # mark = mark[:(classes // 2)] + mark[:(classes // 2)]
    # method 3:
    mark = [mark[0]] * classes

    label = []
    if labels is None:
        for i in range(1, classes // 2 + 1):
            lb = 'S-' + str(i)
            label.append(lb)
        for i in range(1, classes // 2 + 1):
            lb = 'T-' + str(i)
            label.append(lb)
        labels = label
    # print(len(labels), classes)
    assert len(labels) == classes

    set_figure(ms=6., fig_w=5., font_size=8, tick_size=8, lw=1.)
    figs = plt.figure()  # figsize:[6.4, 4.8]
    ax = figs.add_subplot(111)
    for i in range(1, classes + 1):
        # s: 大小 建议50-100, alpha: 不透明度 0.5-0.8
        ax.scatter(da[(i - 1) * shot:i * shot, 0], da[(i - 1) * shot:i * shot, 1], s=30,
                   c=color[(i - 1) * shot:i * shot], alpha=1,
                   marker=mark[i - 1], label=labels[i - 1])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # ax.legend(prop=font, ncol=4, bbox_to_anchor=(0.5, -0.1), loc="upper center")
    ax.legend(ncol=2, loc="upper left").get_frame().set_linewidth(1)

    if name is not None:
        title = 'UMAP embedding of %s (time %.2fs)' % (name, (time() - t0))
        plt.title(title)
    print('t-SNE Done!')
    return figs
