import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import numpy as np
from torch_scatter import scatter_mean
"""
4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
3 个点
无向图
# """

# edge_index = torch.tensor([[0, 1, 1, 2],
#                                 [1, 0, 2, 1]], dtype = torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype = torch.float)

# data = Data(x = x, edge_index = edge_index)

# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# data = Data(x = x, edge_index = edge_index.t().contiguous())
# print(edge_index.t())
# print(edge_index)
# print(edge_index.t().contiguous)
root = 'D:/PythonProject/dl/PYG/Datasets'
dataset = TUDataset(root = root, name = 'ENZYMES')
# data = dataset[0]
# print(data)
# print(len(dataset), type(dataset))
# print(data.x.shape)
# print(data.edge_index.shape)
# print(data.y)

# for i in range(len(dataset)):
#     print(dataset[i].y)

"""
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # 输出 (2, 3)，表示数组是 2x3 的矩阵

import torch

tensor = torch.randn(3, 4)
print(tensor.shape)  # 输出 torch.Size([3, 4])
print(tensor.size())  # 输出 torch.Size([3, 4])

import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
print(df.shape)  # 输出 (3, 2)，表示 DataFrame 是 3 行 2 列

"""

loader = DataLoader(dataset, batch_size=32, shuffle=True)