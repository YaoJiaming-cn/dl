import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
labels = labels.view(-1, 1)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle = True)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

#模型初始化
model = LinearNet(num_inputs)

# for param in model.parameters():
#     print(param)

#定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.03)
# print(optimizer)

# 记录每个 epoch 的损失
losses = []

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in data_iter:
        optimizer.zero_grad()
        outputs =model(inputs)
        loss =criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    losses.append(running_loss / len(data_iter))
    print(f'epoch {epoch + 1}, loss {running_loss / len(data_iter)}')

# 绘制损失曲线
fig = plt.figure(figsize = (15, 8), dpi = 80)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
fig.savefig('Training Loss Curve.png')
