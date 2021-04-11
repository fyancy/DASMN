from abc import ABC

import torch
import torch.nn.functional as F
import torch.nn.modules as nn

Layer8 = True


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )


class Encoder(nn.Module, ABC):
    def __init__(self, cb_num=8):
        super().__init__()
        self.h_dim = 64
        self.z_dim = 64
        self.channel = 1
        print('The Convolution Channel: {}'.format(self.h_dim))
        print('The Convolution Block: {}'.format(cb_num))
        conv1 = conv_block(self.channel, self.z_dim)
        conv_more = [conv_block(self.h_dim, self.z_dim) for i in range(cb_num - 1)]
        self.conv_blocks = nn.Sequential(conv1, *conv_more)

    def forward(self, x):
        feat = self.conv_blocks(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat


class CNN(nn.Module, ABC):
    def __init__(self, nc, DIM, cb_num=8, drop_rate=0.3):
        super().__init__()
        self.prob = drop_rate
        self.encoder = Encoder(cb_num)
        fea_dim = int(64 * DIM / (2 ** cb_num))  # 256
        h_dim = 128
        self.linear1 = nn.Linear(fea_dim, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.linear2 = nn.Linear(h_dim, nc)
        # 一定要把所有layer定义在init函数中，否则backward没有意义

    def get_features(self, x):
        return self.encoder.forward(x)  # [bsize, dim]

    @staticmethod
    def get_loss_acc(out, y):
        prob = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(prob, y)
        pre = torch.max(prob, dim=1)[1]  # (values, indexes)
        acc = torch.eq(pre, y).float().mean()
        return acc, loss

    def forward(self, x, y):
        """

        :param x: [nc, num, 1, 2048]
        :param y: [nc, num]
        :return: acc, loss
        """
        (nc, num) = x.shape[:2]
        x = x.reshape(nc * num, 1, -1)
        y = y.reshape(-1)
        # === Encoder ====
        feature = self.get_features(x)  # [bsize, dim]
        # === Classifier ====
        out = F.dropout(feature, self.prob, self.training)
        out = F.relu(self.bn1(self.linear1(out)))
        out = F.dropout(out, self.prob, self.training)
        out = self.linear2(out)
        acc, loss = self.get_loss_acc(out, y)

        return acc, loss
