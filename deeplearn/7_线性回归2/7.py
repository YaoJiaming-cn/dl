import torch
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2 #特征数
num_examples = 1000 #样本数

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype = torch.float32)
labels = features @ torch.tensor(true_w, dtype = torch.float32)+ true_b
labels += torch.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype = torch.float32)

fig = plt.figure(figsize = (10, 6), dpi = 60)
plt.scatter(features[:, 0].numpy(),labels.numpy())
plt.show()