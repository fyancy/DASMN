from sklearn.preprocessing import StandardScaler, normalize, maxabs_scale
import numpy as np
import torch

device = torch.device('cuda:0')


def my_normalization(x):  # (n, length)
    # method 1:
    x = x - np.mean(x, axis=1, keepdims=True)
    x = maxabs_scale(x, axis=1)  # 效果也很好, 2
    # method 2:
    # x = [(k - np.mean(k)) / (max(k) - min(k)) for k in x]  # 效果好, 3
    # method 3: # 效果 1
    # x = x - np.mean(x, axis=1, keepdims=True)  # 是否去均值 需谨慎！！
    # x = normalize(x, norm='l2', axis=1)
    # method 4:
    # x = [(2*k - max(k) - min(k)) / (max(k) - min(k)) for k in x]  # 效果差, 4
    # print("Normalize data ... Done.")

    return np.asarray(x)


def sample_task_tr(tasks, way, shot, length, chn=1):
    """
    sample a task from tasks for ProtoNet
    :param length:
    :param chn:
    :param way: n-way k-shot
    :param shot: ns or nq
    :param tasks: [nc, n, dim], ns=nq by default.
    :return: [way, n_q, CHN, DIM]
    """

    assert tasks.shape[1] >= shot * 2
    n_s = n_q = shot
    tasks = tasks.reshape(tasks.shape[0], tasks.shape[1], chn, length)

    shuffle_nc = torch.randperm(tasks.shape[0])[:way]  # training
    support = torch.zeros([way, n_s, chn, length], dtype=torch.float32)
    query = torch.zeros([way, n_q, chn, length], dtype=torch.float32)

    for i, cls in enumerate(shuffle_nc):
        selected = torch.randperm(tasks.shape[1])[:n_s + n_q]
        support[i] = tasks[cls, selected[:n_s]]
        query[i] = tasks[cls, selected[n_s:n_s + n_q]]
    support, query = support.to(device), query.to(device)
    return support, query


def sample_task_te(tasks, way, shot, length, chn=1):  # for testing
    """
    sample a task from tasks for ProtoNet
    :param length:
    :param chn:
    :param way: n-way k-shot
    :param shot: ns or nq
    :param tasks: [nc, n, dim], ns=nq by default.
    :return: [way, n_q, CHN, DIM]
    """
    assert tasks.shape[1] >= shot * 2
    n_s = n_q = shot
    tasks = tasks.reshape(tasks.shape[0], tasks.shape[1], chn, length)

    shuffle_nc = torch.randperm(tasks.shape[0])[:way]  # training
    support = torch.zeros([way, n_s, chn, length], dtype=torch.float32)
    query = torch.zeros([way, n_q, chn, length], dtype=torch.float32)

    for i, cls in enumerate(shuffle_nc):
        support[i] = tasks[cls, :n_s]  # 测试时，固定support set
        selected = torch.randperm(tasks.shape[1] - n_s)[:n_q]  # 不考虑 n_s这部分
        query[i] = tasks[cls, selected[:n_q] + n_s]
    support, query = support.to(device), query.to(device)
    return support, query


if __name__ == "__main__":
    select = torch.randperm(10 - 2)[:5]
    print(select)
    print(select + 2)
