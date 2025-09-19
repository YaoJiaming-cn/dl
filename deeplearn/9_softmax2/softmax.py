from torch.utils.data import DataLoader, TensorDataset
import sys
import torchvision
import torch

# 避免 Windows 系统中的多进程并发问题
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不不⽤用额外的进程来加速读取数据
else:
    num_workers = 4

# ~/ 代表用户主目录
# mnist_train = torchvision.datasets.FashionMNIST(root = '~/Datasets/FashionMNIST', train = True, download = True, transform = torchvision.transforms.ToTensor())
# mnist_test = torchvision.datasets.FashionMNIST(root = '~/Datasets/FashionMNIST', train = False, download = True, transform = torchvision.transforms.ToTensor())

mnist_train = torchvision.datasets.FashionMNIST(root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST', train = True, download = True, transform = torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST', train = False, download = True, transform = torchvision.transforms.ToTensor())

batch_size = 256

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)