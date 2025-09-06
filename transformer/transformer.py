import torch
from torch import nn


#定义layerNorm标准化
class LayerNorm(nn.Module):
    def __init__(self, feature,eps = 1e-6):
        """
        :param feature:注意力机制的x的大小
        :param eps:
        """
        super(LayerNorm, self).__init__()
        #定义了三个超参数
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std+self.eps) + self.b_2



# 残差网络标准化层 + 上一层sublayer层
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        #做了layerNorm标准化
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))