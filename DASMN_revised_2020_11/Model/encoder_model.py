"""
yancy F. 2020/10/31
For revised DASMN.
"""

from abc import ABC

import torch
import torch.nn.modules as nn
import torch.nn.functional as F

from my_utils.metric_utils import Euclidean_Distance

device = torch.device('cuda:0')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Conv_CHN = 64
K_SIZE = 3
PADDING = (K_SIZE - 1) // 2


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=K_SIZE, padding=PADDING),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class Encoder(nn.Module, ABC):
    def __init__(self, in_chn=1, cb_num=8):
        super().__init__()
        print('The Convolution Channel: {}'.format(Conv_CHN))
        print('The Convolution Block: {}'.format(cb_num))
        # self.se1 = SELayer(Convolution_CHN)
        conv1 = conv_block(in_chn, Conv_CHN)
        conv_more = [conv_block(Conv_CHN, Conv_CHN) for i in range(cb_num - 1)]
        self.conv_blocks = nn.Sequential(conv1, *conv_more)

    def forward(self, x):
        feat = self.conv_blocks(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat


class MetricNet(nn.Module, ABC):
    def __init__(self, way, ns, nq, vis, cb_num=8):
        super().__init__()
        self.chn = 1
        self.way = way
        self.ns = ns
        self.nq = nq
        self.vis = vis
        self.encoder = Encoder(cb_num=cb_num)
        # self.criterion = nn.CrossEntropyLoss()  # ([n, nc], n)

    def get_loss(self, net_out, target_id):
        # method 1:
        # log_p_y = torch.log_softmax(net_out, dim=-1).reshape(self.way, self.nq, -1)  # [nc, nq, nc]
        # loss_val = -log_p_y.gather(dim=2, index=target_ids).squeeze(dim=-1).reshape(-1).mean()
        # y_hat = log_p_y.max(dim=2)[1]  # [nc, nq]

        # method 2:
        log_p_y = torch.log_softmax(net_out, dim=-1)  # [nc*nq, nc], probability.
        loss = F.nll_loss(log_p_y, target_id.reshape(-1))  # (N, nc), (N,)
        y_hat = torch.max(log_p_y, dim=1)[1]  # [nc*nq]
        acc = torch.eq(y_hat, target_id).float().mean()

        return loss, acc, y_hat, -log_p_y.reshape(self.way, self.nq, -1)

    def get_features(self, x):
        return self.encoder.forward(x)

    def forward(self, xs, xq, sne_state=False):
        # target_ids [nc, nq]
        target_id = torch.arange(self.way).unsqueeze(1).repeat([1, self.nq])
        target_id = target_id.long().to(device)
        # ================
        # x = torch.cat([xs.reshape(self.way * self.ns, self.chn, -1),
        #                xq.reshape(self.way * self.nq, self.chn, -1)], dim=0)
        # z = self.get_features(x)
        # z_proto = z[:self.way * self.ns].reshape(self.way, self.ns, z.shape[-1]).mean(dim=1)
        # zq = z[self.way * self.ns:]
        # =================
        xs = xs.reshape(self.way * self.ns, self.chn, -1)
        xq = xq.reshape(self.way * self.nq, self.chn, -1)
        zs, zq = self.get_features(xs), self.get_features(xq)  # (nc*ns, z_dim)
        z_proto = zs.reshape(self.way, self.ns, -1).mean(dim=1)  # (nc, z_dim)

        dist = Euclidean_Distance(zq, z_proto)  # [nc*ns, nc]
        loss_val, acc_val, y_hat, label_distribution = self.get_loss(-dist, target_id.reshape(-1))

        # if sne_state and self.ns > 1:
        #     self.draw_feature(zq, target_id, y_hat)
        #     self.draw_label(label_distribution, target_id)

        return loss_val, acc_val, zq  # {'loss': loss_val.item(), 'acc': acc_val.item()}


if __name__ == "__main__":
    e = Encoder(cb_num=8)
    data = torch.ones([12, 1, 1024], dtype=torch.float)
    print(e)
    print(e.forward(data).shape)

