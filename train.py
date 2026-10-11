import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# 训练集的数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# 测试集只做标准化，不能做随机增强！
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR10数据集
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self, x):
        return self.net(x)

# 后面换成ResNet18的代码
# model = resnet18(num_classes=10)
# model.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False) #适配32×32图片
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) # LR衰减

# TensorBoard
writer = SummaryWriter("runs/cifar10_exp")

# 早停参数
best_val_acc = 0
patience = 10
early_stop_cnt = 0
max_epoch = 100

for epoch in range(max_epoch):
    # =====训练阶段=====
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # =====验证阶段=====
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)

    # TensorBoard记录
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)

    print(f"Epoch {epoch:3d} | Train Loss:{avg_train_loss:.3f} | Val Acc:{val_acc:.3f}")

    # =====Checkpoint保存 + Early Stopping=====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_cnt = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_val_acc,
        }, "best_cifar10.pth")
        print(f"✅ 保存最优模型，acc={best_val_acc:.3f}")
    else:
        early_stop_cnt += 1
        if early_stop_cnt >= patience:
            print(f"⏹️ 早停触发，连续{patience}轮无提升，结束训练")
            break

writer.close()
# 加载最优权重
checkpoint = torch.load("best_cifar10.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("CIFAR10 Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()