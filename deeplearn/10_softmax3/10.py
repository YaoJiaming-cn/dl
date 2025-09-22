import torch
import torchvision
import numpy as np
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

mnist_train = torchvision.datasets.FashionMNIST(root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST', train = True, download = True, transform = torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST', train = False, download = True, transform = torchvision.transforms.ToTensor())

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不不⽤用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 每个样本输⼊是高和宽均为28像素的图像
# 28x28 =784
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))
# print(X.sum(dim=0))
# print(X.sum(dim=1))

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这⾥里里应用了了广播机制
    # 能够得到合法的概率分布

# X = torch.tensor(np.random.random(10).reshape(2, 5))
# print(X, X.sum(dim = 1))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim = 1))

# torch.nn.Module，会自动有 __call__ 方法
# 调用 model(X) 时会自动转去执行 forward(X)
class SOFTMAX_Model(nn.Module):
    def __init__(self):
        super(SOFTMAX_Model, self).__init__()
    def forward(self, X):
        # torch.mm : matrix multiply（矩阵乘法）
        x = softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
        return x

model = SOFTMAX_Model()

# y_hat = torch.tensor([[0.1, 0.3, 0.6],
#                       [0.4, 0.2, 0.5]])
# y = torch.LongTensor([[0, 1, 0]])
# # 通过使⽤用gather 函数，我们得到了了2个样本的标签的预测概率
# # dim = 1 沿着列的方向选，遍历所有行
# # dim = 0 沿着行的方向选，遍历所有列
# y_gather = y_hat.gather(0, y)
# print(y_gather)

y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.4, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
def cross_entropy_prob(y_hat, y, eps=1e-12):
    # y_hat 是概率（你前向里有 softmax），这里取每个样本真实类的概率并做 -log
    p_true = y_hat.gather(1, y.view(-1, 1)).clamp_min(eps)
    return -torch.log(p_true).mean()  # 返回标量 loss

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().sum().item()


# print(cross_entropy(y_hat, y))
# def accuracy(y_hat, y):
#     # argmax : 沿着某个维度找最大值的下标
#     """
#     :return: y_hat.argmax(dim=1) == y的结果是形状为[batch_size]的tensor，
#             每一个元素是布尔，比如tensor([False,  True])
#     """
#     return (y_hat.argmax(dim=1) == y).float().mean().item()
# print(accuracy(y_hat, y))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# print(evaluate_accuracy(test_iter, model))

num_epochs = 5
lr = 0.1
optimizer = optim.SGD([W, b], lr=0.1)

def train_softmax(net,
                  train_iter,
                  test_iter,
                  loss_fn,
                  num_epochs,
                  optimizer,
                  accuracy):
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)                    # 前向（已是概率）
            loss = loss_fn(y_hat, y)          # 标量

            loss.backward()
            optimizer.step()

            # 统计
            batch_size_cur = y.size(0)
            running_loss += loss.item() * batch_size_cur
            correct += accuracy(y_hat, y)
            total += batch_size_cur

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(test_iter, net)

        print(f'epoch {epoch+1}, loss {train_loss:.4f}, '
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')

train_softmax(model,
              train_iter,
              test_iter,
              cross_entropy_prob,
              num_epochs,
              optimizer,
              accuracy)