import sys
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# ===================== 1) Data =====================
def get_fashion_mnist_loaders(batch_size=256, resize=None):
    tfms = []
    if resize is not None:
        tfms.append(transforms.Resize(resize))
    tfms.append(transforms.ToTensor())
    transform = transforms.Compose(tfms)

    root = 'D:/PythonProject/dl/deeplearn/Datasets/FashionMNIST'  # 你的本地路径
    train_ds = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_ds  = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_iter  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

# ===================== 2) Model =====================
num_inputs = 28 * 28
num_outputs = 10

class FlattenLayer(nn.Module):
    def forward(self, x):  # x: (B, 1, 28, 28)
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_outputs)
)

# 初始化：权重 ~ N(0, 0.01), 偏置=0
init.normal_(net[1].weight, mean=0.0, std=0.01)
init.constant_(net[1].bias, 0.0)

# ===================== 3) Loss & Optimizer =====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# ===================== 4) Metrics =====================
@torch.no_grad()
def evaluate_accuracy(data_iter, net, device=None):
    net.eval()
    if device is None:
        device = next(net.parameters()).device
    correct, total = 0, 0
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        logits = net(X)                 # shape: (B, 10)
        pred = logits.argmax(dim=1)     # shape: (B,)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# ===================== 5) Train Loop =====================
def train_softmax(net, train_iter, test_iter, loss_fn, num_epochs, optimizer, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        running_loss, correct, total = 0.0, 0, 0

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = net(X)                  # 未经过 softmax 的分数
            loss = loss_fn(logits, y)        # CrossEntropyLoss 直接吃 logits
            loss.backward()
            optimizer.step()

            # 统计
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc  = correct / total
        test_acc   = evaluate_accuracy(test_iter, net, device)
        print(f'epoch {epoch+1}, loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

# ===================== 6) Run =====================
if __name__ == "__main__":
    batch_size = 256
    num_epochs = 5
    train_iter, test_iter = get_fashion_mnist_loaders(batch_size=batch_size)
    train_softmax(net, train_iter, test_iter, criterion, num_epochs, optimizer)
