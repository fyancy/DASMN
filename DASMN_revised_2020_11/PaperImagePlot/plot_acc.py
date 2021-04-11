import matplotlib.pyplot as plt

C01 = [
    [65.65, 74.55, 75.20, 77.00],  # SVM
    [71.20, 76.10, 72.70, 74.05],  # TCA
    [71.55, 86.40, 87.35, 92.10],  # CNN
    [81.70, 94.50, 96.60, 98.10],  # DaNN
    [95.03, 95.97, 96.78, 98.94],  # ProtoNets
    [99.07, 100.0, 100.0, 100.0],  # DASMN
]
Csq = [
    [60.83, 62.67, 63.00, 63.00],  # SVM
    [63.50, 60.17, 51.67, 41.17],  # TCA
    [33.83, 33.33, 46.50, 46.17],  # CNN
    [74.67, 65.17, 45.50, 39.33],  # DaNN
    [63.00, 80.42, 83.93, 88.17],  # ProtoNets
    [99.73, 99.89, 100, 100],  # DASMN
]
Csa = [
    [33.33, 33.33, 33.33, 33.33],
    [33.33, 33.33, 33.33, 33.33],
    [33.33, 35.67, 36.80, 34.50],
    [63.17, 45.50, 58.33, 41.67],
    [61.11, 65.04, 69.44, 70.51],
    [77.13, 86.36, 91.53, 92.44],
]
# Ctrl + Alt + L：格式化代码(与QQ锁定热键冲突，关闭QQ的热键)
CW = [
    [
        65.65, 74.55, 75.20, 77.00,
        63.80, 72.85, 74.10, 75.15,
        62.10, 67.45, 69.00, 69.85,
        72.20, 76.40, 78.00, 80.45,
        70.05, 73.45, 74.80, 78.30,
        68.55, 70.65, 70.35, 70.65,
    ],  # SVM
    [
        71.20, 76.10, 72.70, 74.05,
        65.20, 73.45, 74.90, 74.80,
        63.25, 71.00, 74.05, 70.40,
        78.35, 71.55, 74.20, 81.05,
        76.80, 77.67, 68.15, 63.50,
        68.95, 64.35, 64.40, 66.45,
    ],  # TCA
    [
        71.55, 86.40, 87.35, 92.10,
        68.10, 87.45, 92.25, 92.10,
        64.45, 72.30, 78.35, 78.90,
        71.55, 81.05, 83.35, 82.25,
        63.55, 74.00, 73.45, 75.05,
        68.35, 74.50, 76.30, 77.00,
    ],  # CNN
    [
        81.70, 94.50, 96.60, 98.10,
        79.55, 89.70, 95.60, 97.60,
        72.70, 77.20, 80.20, 81.45,
        82.88, 88.30, 90.15, 91.15,
        70.05, 76.80, 77.50, 79.15,
        71.30, 75.75, 77.35, 78.15,
    ],  # DaNN
    [
        95.03, 95.97, 96.78, 98.94,
        93.97, 95.07, 95.63, 96.83,
        90.31, 91.15, 93.35, 95.61,
        94.93, 95.95, 97.78, 98.85,
        90.08, 92.10, 96.14, 98.41,
        85.26, 89.94, 90.97, 92.73,
    ],  # ProtoNets
    [
        99.07, 100, 100, 100,
        98.41, 99.91, 99.94, 99.97,
        98.33, 99.71, 99.91, 99.93,
        98.39, 99.69, 99.95, 100,
        95.38, 98.67, 99.51, 99.73,
        92.17, 96.97, 98.45, 99.15,
    ],  # DASMN
]

# ablation study
C01_ablation = [
    [83.53, 95.45, 98.10, 99.55],  # advCNN
    [78.24, 92.19, 99.31, 99.40],  # ProtoNets 2
    [99.07, 100.0, 100.0, 100.0],  # DASMN
    [71.55, 86.40, 87.35, 92.10],  # CNN
    [95.03, 95.97, 96.78, 98.94],  # ProtoNets
]
Csq_ablation = [
    [44.17, 64.50, 60.83, 66.33],
    [85.69, 83.31, 72.87, 80.26],
    [99.73, 99.89, 100.0, 100.0],
    [33.83, 33.33, 46.50, 46.17],
    [63.00, 80.42, 83.93, 88.17],
]

Csa_ablation = [
    [51.50, 65.00, 69.00, 74.00],
    [60.24, 60.76, 63.53, 65.93],
    [77.13, 86.36, 91.53, 92.44],
    [33.33, 35.67, 36.80, 34.50],
    [61.11, 65.04, 69.44, 70.51],
]

CW_ablation = [
    [
        83.53, 95.45, 98.10, 99.55,
        88.45, 99.10, 91.10, 99.00,
        77.55, 90.60, 97.75, 97.97,
        86.20, 95.65, 98.95, 99.70,
        71.45, 90.05, 97.90, 97.40,
        76.10, 93.25, 96.05, 97.40,
    ],  # advCNN
    [
        78.24, 92.19, 99.31, 99.40,
        82.52, 97.80, 99.93, 99.86,
        79.41, 97.48, 95.33, 96.49,
        89.90, 96.85, 98.11, 99.45,
        89.42, 91.94, 94.63, 95.75,
        89.58, 91.45, 91.24, 95.34,
    ],  # ProtoNets 2
    [
        99.07, 100, 100, 100,
        98.41, 99.91, 99.94, 99.97,
        98.33, 99.71, 99.91, 99.93,
        98.39, 99.69, 99.95, 100,
        95.38, 98.67, 99.51, 99.73,
        92.17, 96.97, 98.45, 99.15,
    ],  # DASMN
]


# ============================ plot

def set_figure(font_size=10., tick_size=8., ms=7., lw=1.2, fig_w=8.):
    # print(plt.rcParams.keys())  # 很有用，查看所需属性
    # exit()
    cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
    w = fig_w * cm_to_inc  # cm ==> inch
    h = w * 3 / 4
    plt.rcParams['figure.figsize'] = (w, h)  # 单位 inc
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['figure.figsize'] = (14 * cm_to_inc, 6 * cm_to_inc)

    # 1. Times New Roman:
    plt.rc('font', family='Times New Roman', weight='normal', size=float(font_size))
    # 2. Helvetica:
    # font = {'family': 'sans-serif', 'sans-serif': 'Helvetica',
    #         'weight': 'normal', 'size': float(font_size)}
    # plt.rc('font', **font)  # pass in the font dict as kwargs

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


marker_style = ['o', '^', 'v', 's', 'd', '>', '<', 'h']
marker_color = [
    [0.00, 0.45, 0.74],  # 蓝色
    [0.93, 0.69, 0.13],  # 黄色
    [0.85, 0.33, 0.10],  # 橘红色
    [0.49, 0.18, 0.56],  # 紫色
    [0.47, 0.67, 0.19],  # 绿色
    [0.30, 0.75, 0.93],  # 青色
    [0.64, 0.08, 0.18],  # 棕色
]
line_style = ['-', '--', '-.', ':']


def plot_acc(data, x_lim, y_lim, legend_list, save_path):
    lw = 1.2
    ms = 7
    set_figure(lw=lw, ms=ms, font_size=10, tick_size=10)
    plt.figure(1)
    x = [1, 2, 3, 4]
    x_name = ['10', '30', '50', '100']

    for i in range(len(data)):
        plt.plot(x, data[i], lw=lw, color='k', marker=marker_style[i],
                 markerfacecolor=marker_color[i], markeredgecolor='k')
    # adjust
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Sample size')
    plt.xticks(x, x_name)
    # plt.yticks([0, 20, 40, 60, 80, 95])
    # plt.legend(legend_list, ncol=2, loc="lower center",
    #            labelspacing=0.2).get_frame().set_linewidth(lw)
    plt.legend(legend_list, ncol=2, loc="lower center").get_frame().set_linewidth(lw)

    # save
    order = input('Save fig' + save_path[-4:] + '? Y/N\n')
    if order == 'y' or order == 'Y':
        # plt.savefig(save_path, dpi=300, format='svg', bbox_inches='tight')
        plt.savefig(save_path + '.svg', dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
        plt.savefig(save_path + '.png', dpi=600, format='png', bbox_inches='tight', pad_inches=0.01)
        # 保留0.01白边，防止切掉线框.注意剪裁之后，图像尺寸会变小。

    # show
    plt.show()


def plot_box_acc(data, legend_list, save_path):
    lw = 1.2
    ms = 7
    # set_figure(font_size=10, tick_size=8, lw=lw, ms=ms, fig_w=8.5)
    set_figure(font_size=10, tick_size=10, lw=lw, ms=ms, fig_w=8)
    plt.figure(1)

    plt.boxplot(data, showmeans=True, showfliers=False,
                labels=legend_list,
                meanprops=dict(markersize=6), whiskerprops=dict(linewidth=lw),
                boxprops=dict(linewidth=lw),
                capprops=dict(linewidth=lw),  # 首尾横线属性
                medianprops=dict(linewidth=lw),  # 中位线属性
                )
    # plt.violinplot(data, showmeans=True, showmedians=True)
    plt.ylabel('Accuracy (%)')
    # plt.xlabel('Models')
    # save
    order = input('Save fig' + save_path[-4:] + '? Y/N\n')
    if order == 'y' or order == 'Y':
        plt.savefig(save_path + '.svg', dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
        plt.savefig(save_path + '.png', dpi=600, format='png', bbox_inches='tight', pad_inches=0.01)
        # 保留0.01白边，防止切掉线框.注意剪裁之后，图像尺寸会变小。

    plt.show()


def plot_acc_ablation(data, x_lim, y_lim, legend_list, save_path):
    lw = 1.2
    ms = 7
    set_figure(lw=lw, ms=ms, font_size=10, tick_size=10)
    plt.figure(1)
    x = [1, 2, 3, 4]
    x_name = ['10', '30', '50', '100']

    for i in range(2):
        plt.plot(x, data[i], lw=lw, color='k', marker=marker_style[i],
                 markerfacecolor=marker_color[i], markeredgecolor='k')

    plt.plot(x, data[2], lw=lw, color='k', marker=marker_style[5],
             markerfacecolor=marker_color[5], markeredgecolor='k')  # DASMN
    plt.plot(x, data[3], '--', lw=lw, color='k', marker=marker_style[0],
             markerfacecolor=marker_color[3], markeredgecolor='k')  # CNN
    plt.plot(x, data[4], '--', lw=lw, color='k', marker=marker_style[1],
             markerfacecolor=marker_color[4], markeredgecolor='k')  # ProtoNets
    # adjust
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Sample size')
    plt.xticks(x, x_name)
    # plt.yticks([0, 20, 40, 60, 80, 95])
    plt.legend(legend_list, ncol=2, loc="lower center",
               labelspacing=0.2).get_frame().set_linewidth(lw)

    # save
    order = input('Save fig' + save_path[-4:] + '? Y/N\n')
    if order == 'y' or order == 'Y':
        # plt.savefig(save_path, dpi=300, format='svg', bbox_inches='tight')
        plt.savefig(save_path + '.svg', dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
        plt.savefig(save_path + '.png', dpi=600, format='png', bbox_inches='tight', pad_inches=0.01)
        # 保留0.01白边，防止切掉线框.注意剪裁之后，图像尺寸会变小。

    # show
    plt.show()


if __name__ == '__main__':
    leg_list = ['SVM', 'TCA', 'CNN', 'DaNN', 'ProtoNets', 'DASMN']
    fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\Csq'
    plot_acc(C01, x_lim=[0.8, 4.2], y_lim=[50, 103], legend_list=leg_list, save_path=fig_path)
    exit()
    # plot_acc(Csq, x_lim=[0.8, 4.2], y_lim=[0, 105], save_path=fig_path, legend_list=leg_list)
    # fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\Csa'
    # plot_acc(Csa, x_lim=[0.8, 4.2], y_lim=[-10, 97], save_path=fig_path, legend_list=leg_list)

    # box plot
    # fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\CW_box'
    # plot_box_acc(CW, legend_list=leg_list, save_path=fig_path)

    # Ablation study
    # leg_list = ['advCNN', 'ProtoNets Ⅱ', 'DASMN', 'CNN', 'ProtoNets']
    # fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\C01_ablation'
    # plot_acc_ablation(C01_ablation, x_lim=[0.8, 4.2], y_lim=[50, 103], save_path=fig_path, legend_list=leg_list)

    # fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\Csq_ablation'
    # plot_acc_ablation(Csq_ablation, x_lim=[0.8, 4.2], y_lim=[-13, 105], save_path=fig_path, legend_list=leg_list)

    # fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\Csa_ablation'
    # plot_acc_ablation(Csa_ablation, x_lim=[0.8, 4.2], y_lim=[-13, 102], save_path=fig_path, legend_list=leg_list)

    # box plot
    leg_list = ['advCNN', 'ProtoNets Ⅱ', 'DASMN']
    fig_path = r'C:\Users\Asus\Desktop\MyWork\paper_DASMN\DASMN_KBS_revised\figure\case_ACC\CW_ablation_box'
    plot_box_acc(CW_ablation, legend_list=leg_list, save_path=fig_path)
