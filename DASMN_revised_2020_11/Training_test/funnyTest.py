import torch
import numpy as np

# a = torch.tensor([
#     [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
#     [[4, 4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6], [7, 7, 7, 7, 7]],
#     [[8, 8, 8, 8, 8], [9, 9, 9, 9, 9], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
# ])  # shape: (3, 4, 5)

#
# a = np.array([1, 2, 3])  # 3
# b = np.array([[1], [1], [1]])
# print(np.matmul())
exit()
color_set = [
        [0.00, 0.45, 0.74],  # 蓝色
        [0.93, 0.69, 0.13],  # 黄色
        [0.85, 0.33, 0.10],  # 橘红色
        [0.49, 0.18, 0.56],  # 紫色
        [0.47, 0.67, 0.19],  # 绿色
        [0.30, 0.75, 0.93],  # 青色
        [0.64, 0.08, 0.18],  # 棕色
    ]
classes = 6
color = color_set[:classes//2] + color_set[:classes//2]
color = np.asarray(color)
color = np.tile(color[:classes][:, None], (1, 2, 1)).reshape(-1, 3)

print(color.shape)
print(color)

