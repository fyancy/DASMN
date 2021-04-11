import torch


def Euclidean_Distance(x, p):
    """
    :param x: n x d   N-example, P-dimension for each example; zq
    :param p: nc x d   M-Way, P-dimension for each example, but 1 example for each Way; z_proto
    :return: [n, nc]
    """
    x = x.unsqueeze(dim=1)  # [n, d]==>[n, 1, d]
    p = p.unsqueeze(dim=0)  # [nc, d]==>[1, nc, d]
    return torch.pow(x - p, 2).mean(dim=2)


