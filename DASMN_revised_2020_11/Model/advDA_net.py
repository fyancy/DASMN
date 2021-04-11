from abc import ABC

import torch
from torch.autograd import Function
import torch.nn.modules as nn


class GRL(Function):
    """
    Implement the Gradient Reversal Layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

    @staticmethod
    def grad_reverse(x, constant):
        return GRL.apply(x, constant)


class DomainClassifier(nn.Module, ABC):
    """
    a simple fully-connected network. Linear;BatchNorm;ReLU;Linear;
    """

    def __init__(self, DIM):
        super().__init__()
        NUM_BLOCK = 8
        FEATURE_CHN = 64
        x_dim = int(FEATURE_CHN * (DIM // 2 ** NUM_BLOCK))  # 2048:8192; 1024:4096
        feature = 256  # 100(original), 64(CW2SQ), 32, 16. choose: 256, 16
        # print('The NUM of ConvBlocK: {}'.format(NUM_BLOCK))
        print('The FC features: {}\n'.format(feature))
        self.create_feat = nn.Linear(x_dim, feature)  # weight shape: (feature, x_dim)
        self.discriminator = nn.Sequential(nn.BatchNorm1d(feature), nn.ReLU(),
                                           nn.Linear(feature, 2))
        # self.simpleLayer = nn.Sequential(nn.Linear(x_dim, feature), nn.BatchNorm1d(feature), nn.ReLU(),
        #                                  nn.Linear(feature, 2))
        # self.simpleLayer = nn.Sequential(nn.Linear(x_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
        #                                  nn.Linear(128, feature), nn.BatchNorm1d(feature), nn.ReLU(),
        #                                  nn.Linear(feature, 2))

    def forward(self, x, constant):
        out = GRL.grad_reverse(x, constant)
        # out = self.simpleLayer(out)
        feat = self.create_feat(out)
        out = self.discriminator(feat)
        out = torch.log_softmax(out, dim=1)
        return out, feat  # [n, 2]


if __name__ == "__main__":
    a = torch.rand([3, 512])
    d = DomainClassifier(DIM=2048)
    for n, p in d.named_parameters():
        print(n, p.shape)
    print(d)

    exit()
    data = d(a, constant=10)[0]
    print(data)
    print(d)
    print(data.neg())
