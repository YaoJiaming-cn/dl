# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2 #特征数
num_examples = 1000 #样本数

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float32)
labels = (true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b)
labels += torch.tensor(np.random.normal(0, 0.1, size=labels.size()), dtype=torch.float32)


# fig = plt.figure(figsize = (10, 6), dpi = 60)
# plt.scatter(features[:, 0].numpy(),labels.numpy())
# plt.show()

# def set_figsize(figsize = (3.5, 2.5)):
#     plt.rcParams['figure.figsize'] = figsize
#
# set_figsize()
#
# plt.scatter(features[:, 0].numpy(), labels.numpy(), 5)
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size,num_examples)]) # 最后⼀一次可能不不⾜足⼀一个batch
        # 一个函数如果被定义了yield,那么这个函数就是generator function 函数返回值是 generator
        yield  features.index_select(0, j), labels.index_select(0,
j)

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break #break让运行一次就跳出循环，没有遍历整个迭代器



w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)


def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这⾥里里更改param时用的param.data


lr = 0.03
num_epochs = 10
batch_size = 20
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b ), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {train_l.mean().item()}')