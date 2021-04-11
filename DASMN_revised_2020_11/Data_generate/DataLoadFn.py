import numpy as np
import torch

from Data_generate.Data_file.CWdata_dir import T0, T1, T2, T3, T_sq, T_sa
from Data_generate.Data_file.SQdata_dir import sq3_29_0, sq3_29_1, sq7_29_0, sq7_29_1, SQ_data_get
from Data_generate.Data_file.SAdata_dir import SA7_20, SA3_10, SA3_20, SA3_25

from Data_generate.mat2csv import get_data_csv
from my_utils.training_utils import my_normalization

normalization = my_normalization


def sample_shuffle(data):
    """
    required: data.shape [Nc, num, ...]
    :param data: [[Nc, num, ...]]
    """
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    return data


class DataGenFn:
    def __init__(self):
        # CWRU data:
        self.case10 = [T0, T1, T2, T3]  # C01, C02...
        self.case_cross = dict(sq=T_sq, sa=T_sa)  # cw2sq:NC, IF, OF; cw2sa:NC, OF, RoF

        # SQ data:
        self.sq3 = [sq3_29_0, sq3_29_1]  # 29Hz
        # self.sq3 = [sq3_39_0, sq3_39_1]  # 39Hz
        self.sq7 = [sq7_29_0, sq7_29_1]  # 39Hz

        # SA data:
        self.SA7 = [SA7_20]
        self.SA3 = [SA3_25]  # NC, OF, RoF.

    def CW_10way(self, way, order, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        file_dir = [self.case10[order]]
        print('CW_{}way load [{}] loading ……'.format(way, order))
        n_way = len(file_dir[0])  # 10 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, examples)  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def CW_cross(self, way, examples=200, split=30, data_len=2048, shuffle=False,
                 normalize=True, label=False, tgt_set=None):
        # 1. examples each file <= 119 * 1024
        # 2. if examples>=119, the value of overlap should be True
        print('CW_{}way [cw to {}] loading ……'.format(way, tgt_set))
        Class = dict(sq=['NC', 'IF3', 'OF3'], sa=['NC', 'OF3', 'RoF'])
        if tgt_set == 'sa' or tgt_set == 'sq':
            file_dir = [self.case_cross[tgt_set]]
            print(Class[tgt_set])
        else:
            file_dir = None
            print("Please identify the param: tgt_set, 'sa' or 'sq'\n")

        n_way = len(file_dir[0])  # 3 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way, 1
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)

        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, examples)  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def SQ_37way(self, way=3, examples=100, split=30, shuffle=False,
                 data_len=2048, normalize=True, label=False):
        """
        :param shuffle:
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each file
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,split,1,2048], [Nc, split];
        [Nc,examples*2-split,1,2048], [Nc, examples*2-split]

        """
        file_dir = self.sq3
        if way == 3:
            file_dir = self.sq3
        elif way == 7:
            file_dir = self.sq7
        print('SQ_{}way loading ……'.format(way))
        # print(file_dir)
        n_way = len(file_dir[0])  # 3/7 way
        n_file = len(file_dir)  # 2 files
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = SQ_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])
        if shuffle:
            data_set = sample_shuffle(data_set)  # 酌情shuffle, 有的时候需要保持测试集和evaluate一致
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, num_each_way)  # [Nc, num_each_way]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def SA_37way(self, way, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False):
        file_dir = None
        if way == 3:
            file_dir = self.SA3
        elif way == 7:
            file_dir = self.SA7
        print('SA_{}way loading ……'.format(way))
        n_way = len(file_dir[0])  # 3/7 way
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way, 1
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = get_data_csv(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)
            label = label.repeat(1, num_each_way)  # [Nc, num_each_way]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,1024], [Nc, 100]
        else:
            return train_data, test_data

# 备注：
# 采用滑窗采样的方法 num =  (N-length) // win_size + 1


if __name__ == "__main__":
    d = DataGenFn()
    # tr_d, tr_l, te_d, te_l = d.CW_10way(way=10, order=0, examples=200, split=20,
    #                                     normalize=False, data_len=1024, label=True)
    tr_d, tr_l, te_d, te_l = d.CW_cross(way=3, examples=200, split=20, tgt_set='sa',
                                        normalize=False, data_len=1024, label=True)
    # tr_d, tr_l, te_d, te_l = d.SA_37way(label=True, way=3, normalize=False, data_len=1024,
    #                                     examples=200, split=20)
    # tr_d, tr_l, te_d, te_l = d.SQ_37way(label=True, way=3, normalize=False, data_len=1024,
    #                                     examples=200, split=20)
    print(tr_d.shape, tr_l.shape)
    print(tr_d[2, 0, 0, :10], tr_l[:, :3])

