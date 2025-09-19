import torch
import torchvision
from matplotlib import pyplot as plt
import time
import sys

"""
torchvision : 构建机器视觉的
torchvision.datasets : ⼀一些加载数据的函数及常⽤用的数据集接⼝
torchvision.models : 包含常用的模型结构（含预训练模型），例例如AlexNet、VGG、ResNet等；
torchvision.transforms : 常用的图⽚片变换，例如裁剪、旋转等；
torchvision.utils : 其他的⼀一些有用的方法
"""

mnist_train = torchvision.datasets.FashionMNIST(root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST', train = True, download = True, transform = torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST', train = False, download = True, transform = torchvision.transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
print(mnist_train)

feature, label = mnist_train[59999]
# print(feature, label)
# print(len(feature))
# print(feature.shape)
# print(feature.size)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # 创建一个包含多个子图的图像
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # 第一个返回值：fig, 它是一个 matplotlib.figure.Figure 对象
    # 第二个返回值：axs, 它是一个包含所有子图（axes）的数组或列表

    # plt.subplots(nrows, ncols, figsize)返回一个元组 (fig, axs)

    # 遍历每张图片和对应的标签
    for f, img, lbl in zip(figs, images, labels):
        # 将图片转换为28x28的矩阵并显示（Fashion MNIST的图片是28x28的）
        # f.imshow(img.view((28, 28)), cmap='gray')
        f.imshow(img.view((28, 28)))
        # 设置标题为标签值
        f.set_title(lbl)
        # 隐藏坐标轴
        f.axis('off')  # 不显示坐标轴

    # 显示所有的图像
    plt.show()

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))