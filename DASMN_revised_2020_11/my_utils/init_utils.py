import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn


def weights_init0(m):
    # xavier在tanh,sigmoid中表现的很好，但在Relu激活函数中表现的很差, 何凯明提出了针对于relu的初始化方法
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0., 0.02)  # generally choose this, 0.02.
        # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif class_name.find('Linear') != -1:
        nn.init.normal_(m.weight, 0., 0.02)  # std=0.5 vs std=0.02, 0.02 generally. better.
        # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu'). good.


def weights_init1(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
        # print(m.bias.data)


def weights_init2(L):
    if isinstance(L, nn.Conv1d):
        n = L.kernel_size[0] * L.out_channels
        L.weight.data.normal_(mean=0, std=np.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm1d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
        # print(L.bias.data)
    elif isinstance(L, nn.Linear):
        L.weight.data.normal_(0, 0.01)
        if L.bias is not None:
            L.bias.data.fill_(1)


def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"set random seed: {seed}")
    # 下面两项比较重要，搭配使用。以精度换取速度，保证过程可复现。
    # https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = False  # False: 表示禁用
    cudnn.deterministic = True  # True: 每次返回的卷积算法将是确定的，即默认算法。
    # cudnn.enabled = True
