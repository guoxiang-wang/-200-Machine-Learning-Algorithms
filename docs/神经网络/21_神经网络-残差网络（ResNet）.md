---
title: "ResNet"
description: "神经网络 主题下的 ResNet 条目，已重写整理为适合公开分享的版本。"
---

> 这一页把 ResNet 的结构、直觉和训练要点重新组织了一遍，便于查阅和分享。

## 快速理解
- 先抓住网络由什么模块组成。
- 再看它解决了什么类型的问题。
- 最后记住训练时最容易出问题的地方。

## 机制详解

### 1. 传统神经网络的表示

我们先看普通前馈神经网络的基本结构：

假设一个输入为 $x$，希望网络学习一个表示为 $H(x)$ 的映射函数（也就是输出）。

那么传统的神经网络直接学习这个函数：

$y = H(x)$

在每一层中，这个 $H(x)$ 是通过多层线性变换（权重矩阵和偏置）加非线性激活函数得到的。

### 2. 残差网络的关键思路

ResNet 不是直接学习 $H(x)$，而是让网络学习一个**残差函数** $F(x)$，其中：

$F(x) = H(x) - x$

也就是说：

$H(x) = F(x) + x$

于是，输出变成了：

$y = F(x) + x$

这个“加上原始输入”的过程就是所谓的**跳跃连接（Skip Connection）或恒等映射（Identity Mapping）**。

### 3. 为什么学习残差更容易？

从优化角度理解：

- 假设最优的函数 $H(x)$ 接近恒等映射（即 $H(x) \approx x$）

- 那么残差 $F(x) = H(x) - x \approx 0$

- 学习一个接近 0 的函数（即什么都不做）通常比学习完整的 $H(x)$ 要简单得多

这样，残差网络为网络提供了一条**最小努力路径（least effort path）**，有助于稳定训练，减轻深层网络中的梯度消失问题。

### 4. 残差块（Residual Block）数学结构

**输入输出形式**：

设某个残差模块的输入为 $x$，输出为 $y$。

一般残差块的结构为：

$y = F(x, \{W_i\}) + x$

其中：

- $F(x, \{W_i\})$：是表示一系列变换（如卷积、激活、归一化）的函数

- $\{W_i\}$：是该模块中各层的权重参数

- $x$：为输入（将被直接加回去）

**举例：两层卷积的残差块**，令：

$F(x) = W_2 \cdot \sigma(W_1 \cdot x)$

其中：

- $W_1$、$W_2$：两个卷积层的权重

- $\sigma$：激活函数（如 ReLU）

则输出为：

$y = W_2 \cdot \sigma(W_1 \cdot x) + x$

这个结构中，**梯度可以沿着加法中的 **$x$** 直接向前传播到前面层**，即便 $F(x)$ 很复杂。

### 5. 反向传播中的梯度分析（核心数学推导）

设损失函数为 $\mathcal{L}$，输出为：

$y = F(x) + x$

我们关心的是损失函数关于输入 $x$ 的梯度：

$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( \frac{\partial F(x)}{\partial x} + I \right)$

注意：

- $\frac{\partial F(x)}{\partial x}$：是复杂映射的梯度

- $I$：是单位矩阵，对应于 $x$ 在加法中直接传播

这说明了一个非常关键的点：

> **即使 **$\frac{\partial F(x)}{\partial x}$** 变得很小（梯度消失），单位矩阵 **$I$** 的存在仍然保证了梯度能稳定地流向前层。**

这就是为什么 ResNet 在非常深的网络中也能训练稳定的数学依据。

### 6. 维度匹配问题与变换

在一些情况下，$F(x)$ 的输出维度与 $x$ 不一致，不能直接相加。为了解决这个问题，可以使用一个线性变换 $W_s$ 对 $x$ 进行升维或降维：

$y = F(x) + W_s \cdot x$

- $W_s$：一般是 1×1 的卷积，用于调整维度

- 如果维度一致，则 $W_s$ 是单位映射

这个处理确保了加法操作在张量维度上合法。

### 7. 整体算法流程（训练阶段）

1. **输入图像数据** $x$

2. **前向传播**：

    - 每个残差块执行 $y = F(x) + x$

    - 如果维度不一致，执行 $y = F(x) + W_s \cdot x$

3. **非线性激活**（如 ReLU）

4. **重复残差块**若干次（可达几十、上百层）

5. **全连接层、Softmax 层**输出分类结果

6. **计算损失函数** $\mathcal{L}$

7. **反向传播计算梯度**，梯度沿主路径和快捷路径传播

8. **参数更新**（使用 SGD、Adam 等优化器）

### 8. 要点归纳

ResNet 的核心是：**从学习映射 **$H(x)$**，转为学习残差函数 **$F(x) = H(x) - x$

每个残差块输出为：

$y = F(x) + x$

**反向传播中梯度公式为**：

$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( \frac{\partial F(x)}{\partial x} + I \right)$

这保证了梯度可以顺利传回前层，避免梯度消失

当维度不匹配时，使用线性映射 $W_s$ 调整：

$y = F(x) + W_s \cdot x$

这种设计让 ResNet 能在保持高精度的同时，将网络深度推到上百甚至上千层，极大地推动了深度学习的发展。

## 完整例子

#### 数据集

这个数据集，大家很熟悉了。

CIFAR-10 是一个包含 10 类彩色图像的小型数据集，每类 6000 张图像，共计 60000 张图像。每张图片大小为 32x32，常用于图像分类模型的入门实验。

### 代码实现

```Python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import os
import random

# 设置随机种子确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 残差块定义
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# ResNet 模型定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 构建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 训练函数
def train_model(num_epochs=30):
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct / total
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)

        # 测试过程
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(testloader.dataset)
        test_acc = correct / total
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return train_loss_list, test_loss_list, train_acc_list, test_acc_list

train_loss, test_loss, train_acc, test_acc = train_model()

# 可视化训练过程
def plot_metrics(train_loss, test_loss, train_acc, test_acc):
    epochs = range(1, len(train_loss) + 1)
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", color="red")
    plt.plot(epochs, test_loss, label="Test Loss", color="blue")
    plt.title("Loss over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy", color="green")
    plt.plot(epochs, test_acc, label="Test Accuracy", color="orange")
    plt.title("Accuracy over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(train_loss, test_loss, train_acc, test_acc)

# 混淆矩阵分析
def plot_confusion_matrix():
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="coolwarm")
    plt.title("Confusion Matrix - ResNet on CIFAR-10", fontsize=16)
    plt.show()

plot_confusion_matrix()
```

#### 1. 数据加载与增强

- `RandomHorizontalFlip`：模拟不同角度的图像

- `RandomCrop`：增加鲁棒性，避免过拟合

- `Normalize`：数据标准化，提高训练速度

#### 2. 残差模块 `BasicBlock`

- 两层卷积 \+ 批归一化 \+ ReLU

- 残差连接 `out += identity` 实现跳跃连接

#### 3. ResNet 模型结构

- 模拟 ResNet-18 的结构 `[2, 2, 2, 2]`：4 个阶段，每阶段 2 个残差块

- 输入层卷积不使用大核（如7x7）以适应 CIFAR-10 的小尺寸图像

- 使用 `AdaptiveAvgPool2d` 将特征图统一为 1x1，再扁平化处理

#### 4. 损失函数与优化器

- 使用 `CrossEntropyLoss` 处理多分类问题

- 使用 `Adam` 优化器，提高收敛速度

- 使用 `StepLR` 学习率调度器，使训练更加稳定

#### 5. 可视化

- 使用 `matplotlib` 和 `seaborn` 绘制：

    - 损失曲线（红\+蓝）

    - 准确率曲线（绿\+橙）

    - 混淆矩阵（coolwarm 热力色系）

#### 6. 混淆矩阵分析

- 查看哪些类别被混淆（如猫狗、车船）

- 帮助进一步改进模型或数据增强策略

### 算法优化建议

**1. 深层网络改进**

可尝试更深层网络，例如：

```Python
ResNet(BasicBlock, [3, 4, 6, 3])  # 类似 ResNet-34
```

**2. 正则化增强**

加入 Dropout 防止过拟合：

```Python
self.dropout = nn.Dropout(0.3)
```

放在全连接层前：

```Python
x = self.dropout(x)
x = self.fc(x)
```

**3. Label Smoothing**

使用标签平滑减少过拟合：

```Python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_prob)
        true_dist.fill_(self.smoothing / (log_prob.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=-1))
```

**4. 数据增强进一步优化**

可引入 Cutout、Mixup、AutoAugment 等高级策略提升泛化能力。

**5. 高效推理**

部署时考虑：

- 模型剪枝

- 量化

- 使用 `torch.jit` 编译模型加速推理

整体上，ResNet 通过跳跃连接成功解决了深层网络难以训练的问题。

## 模型分析

### ResNet 优缺点分析

**优点**

1. **有效缓解梯度消失问题**：残差连接（skip connection）允许梯度直接从输出层反向传播到更浅层，即使网络非常深，也能保持稳定训练。这使得 ResNet 成为第一个能稳定训练上百层网络的架构。

2. **提升模型的表达能力而不过拟合**：ResNet 允许我们在深度增加的同时保持较低的训练误差，而不会因为“模型太复杂”而严重过拟合。这是因为每一层只学习“残差”，即必要的补充信息。

3. **在图像任务中表现优异，适用于小图像任务（如 CIFAR）和大图像任务（如 ImageNet）**：ResNet 是许多后续模型（如 Faster R-CNN、YOLOv5、ResNeXt、Mask R-CNN 等）的基础骨干网络，在工业界和研究中应用广泛。

4. **结构可扩展性强**：可以根据实际需求，选用不同深度的变体，如 ResNet-18、34、50、101、152，灵活性强，适配不同计算资源和任务难度。

5. **训练更快速、更稳定**：在使用相同优化器与超参数条件下，ResNet 通常能更快收敛，并达到更高的准确率。

**缺点**

1. **计算资源需求大**：深度结构意味着参数多，计算量大，显存占用较高，尤其是 ResNet-50 以上版本，难以在低端设备上运行。

2. **结构设计仍需人为干预**：ResNet 虽然解决了深度训练问题，但结构仍需手动配置，难以自动调整；在特定场景下可能需要定制化设计（例如嵌入式设备）。

3. **残差连接非万能**：并不是所有任务引入残差结构都能显著提升效果，特别是在一些浅层任务或对低计算复杂度要求严格的系统中，ResNet 的优势不明显。

4. **低层信息可能被掩盖**：在非常深的网络中，虽然残差连接缓解了信息损失，但仍可能出现低层特征在融合过程中被“稀释”的问题。

### 与相似算法对比分析

|特征|ResNet|VGGNet|DenseNet|EfficientNet|MobileNet|
|---|---|---|---|---|---|
|**结构深度**|深（18~152层）|较深（16、19层）|更深且密集连接|自动搜索的最优结构|轻量级|
|**引入残差/连接机制**|残差连接（Skip）|无|密集连接（Dense）|有效通道利用|Depthwise卷积|
|**参数数量**|较少（比VGG少）|多|参数共享较好|自动优化，参数最优|极少<br>|
|**训练难度**|适中（适合深层训练）|简单（但收敛慢）|训练复杂但效果好|相对较难调参|非常轻便，易训练|
|**适合任务**|通用图像分类、检测等|简单图像识别|医学图像、细粒度分类|高精度大规模图像任务|移动端图像处理|
|**精度表现（CIFAR-10）**|极高（90%以上）|中等（85%左右）|极高（可达92%）|极高|一般（80%左右）|
|**设备适应性**|需要中高性能GPU|可在普通CPU运行|中等需求|高要求|可在手机运行|
|**是否适合迁移学习**|是，常用作主干网络|是，但使用率下降|是，但特征冗余风险|是|是|

### 使用 ResNet 的优选场景

1. **希望使用深层网络且训练稳定**：例如在 CIFAR、ImageNet、医疗图像等任务中，ResNet 能处理数十层甚至上百层的结构，不容易出现梯度消失。

2. **需要高精度分类结果**：在追求分类性能的任务中（例如人脸识别、工业缺陷检测），ResNet 提供较高的表现能力。

3. **迁移学习任务中作为主干特征提取网络**：如使用 ResNet-50 作为 Faster R-CNN 的 backbone，在目标检测任务中表现优异。

4. **用于分析任务中的可解释性增强**：因为残差连接使得梯度流动更清晰，在某些可解释性分析中（例如 Grad-CAM），ResNet 更适合作为分析基础网络。

### 考虑其他算法的场景

1. **资源受限（如移动端、嵌入式）**：推荐使用 MobileNet、ShuffleNet 或 SqueezeNet，此类模型参数小、推理速度快，更适合低功耗设备。

2. **图像之间特征高度重复或需要更强特征融合能力时**：DenseNet 可能优于 ResNet，因为其层与层之间特征是直接相连的，特征复用率更高。

3. **自动化结构优化（NAS）需求时**：可选择 EfficientNet 等基于神经架构搜索的模型，具有更强的结构优化能力。

4. **浅层网络也可达标时**：如某些小型图像数据集（MNIST、Fashion-MNIST），VGG、LeNet 等浅层网络即可胜任，使用 ResNet 反而会增加不必要的计算开销。

## 总结

残差网络是一种通过引入**跳跃连接**来缓解深层神经网络训练困难的问题的架构。它不直接学习最后映射，而是学习输入与输出之间的“残差”，从而提升训练稳定性和模型深度。

ResNet 在图像分类、检测等任务中表现出色，是现代深度学习中最常用的基础结构之一。