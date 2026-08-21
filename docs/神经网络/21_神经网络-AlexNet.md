---
title: "AlexNet"
description: "神经网络 主题下的 AlexNet 条目，已重写整理为适合公开分享的版本。"
---

> 这一页把 AlexNet 的结构、直觉和训练要点重新组织了一遍，便于查阅和分享。

## 快速理解
- 先抓住网络由什么模块组成。
- 再看它解决了什么类型的问题。
- 最后记住训练时最容易出问题的地方。

## 机制详解

#### 全局概览：AlexNet 是什么（用数学对象表述）

任务：多分类（ImageNet 1000 类）。输入图像 $x\in\mathbb{R}^{H\times W\times C}$，输出类别概率 $p\in\Delta^{K-1}$（$K=1000$）。

模型：一个由**卷积层（Conv）→ 非线性（ReLU）→ 归一化（LRN）→ 池化（Pool）→ 全连接（FC）→ Softmax**组成的复合函数

$f_\theta(x)=\mathrm{Softmax}\circ \mathrm{FC}_8\circ \mathrm{Drop}\circ \mathrm{ReLU}\circ \mathrm{FC}_7\circ \mathrm{Drop}\circ \mathrm{ReLU}\circ \mathrm{FC}_6\circ \cdots \circ \mathrm{Conv}_1(x)$

参数 $\theta$ 包含所有卷积核、全连接权重与偏置。

> 常用的 AlexNet 张量尺寸（单卡视角、不考虑原论文的双 GPU 分组计算细节）：输入 $227\times 227\times 3$。

### 1. 各模块的数学定义与尺寸计算

#### 1.1 卷积层（Conv）

**前向：** 给定第 $l$ 层输入特征图 $X^{(l)}\in\mathbb{R}^{H\times W\times C_{in}}$，卷积核 $W^{(l)}\in\mathbb{R}^{k_h\times k_w\times C_{in}\times C_{out}}$，偏置 $b^{(l)}\in\mathbb{R}^{C_{out}}$，步幅 $s$，填充 $p$：
$Z^{(l)}_{i,j,c} =b^{(l)}_c+\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}\sum_{d=0}^{C_{in}-1}W^{(l)}_{u,v,d,c}\cdot X^{(l)}_{i\cdot s+u-p, j\cdot s+v-p, d}$

输出尺寸：

$H'=\Big\lfloor\frac{H-k_h+2p}{s}\Big\rfloor+1,\quad
W'=\Big\lfloor\frac{W-k_w+2p}{s}\Big\rfloor+1,\quad C'=C_{out}.$

**AlexNet 关键设置：**

- Conv1：$k=11\times11$, $s=4$, $p=0$，输入 $227\times227\times3$ → 输出 $55\times55\times96$。

- Conv2：$k=5\times5$, $s=1$, $p=2$ → $27\times27\times256$（在池化后续尺寸基础上）。

- Conv3/4/5：$k=3\times3$, $s=1$, $p=1$。

#### 1.2 非线性激活（ReLU）

- **前向：** $Y=\mathrm{ReLU}(Z)=\max(0,Z)$（逐元素）。

- **性质：** 避免梯度消失、收敛快；导数简单（见反向传播部分）。

#### 1.3 局部响应归一化（LRN, Local Response Normalization）

- 用于模拟“侧抑制”，对同一空间位置在通道维度做归一化。对通道 $i$ 的激活 $a^i_{x,y}$：

$b^i_{x,y} \;=\; \frac{a^i_{x,y}}{\Big(k + \frac{\alpha}{n}\sum_{j=i-\lfloor n/2\rfloor}^{i+\lfloor n/2\rfloor}\big(a^j_{x,y}\big)^2\Big)^{\beta}}$

- AlexNet 常用超参：$k=2,\ \alpha=10^{-4},\ \beta=0.75,\ n=5$。

#### 1.4 池化（Max Pooling）

- **前向（最大池化）：** 在每个通道上以窗口 $k_h\times k_w$，步幅 $s$：

$Y_{i,j,c} \;=\; \max_{0\le u<k_h,\ 0\le v<k_w} X_{i\cdot s+u,\; j\cdot s+v,\; c}.$

- AlexNet 使用 $3\times3$ 窗口、步幅 $2$。

#### 1.5 全连接（Fully-Connected, FC）

- 将上一层张量展平为向量 $x\in\mathbb{R}^{D}$，

$z = Wx + b,\quad W\in\mathbb{R}^{m\times D},\ b\in\mathbb{R}^{m}.$

#### 1.6 Dropout

- 训练时以概率 $p$ 将神经元置零（保留率 $q=1-p$），保留的神经元按 $1/q$ 缩放，期望不变：

$\tilde{x}= \frac{m\odot x}{q},\quad m_i\sim\mathrm{Bernoulli}(q).$

- 推理时不使用掩码与缩放（或采用“训练时不缩放、推理时乘 $q$”的等价实现）。

#### 1.7 Softmax 与交叉熵

- **Softmax：** 给定分类打分（logits）$s\in\mathbb{R}^{K}$：

$p_k = \frac{e^{s_k}}{\sum_{j=1}^K e^{s_j}}$

- **损失（单样本 one-hot 标签 **$y$**）：**

$\mathcal{L} = -\sum_{k=1}^K y_k\log p_k.$

- **带权重衰减（**$L_2$** 正则）：** 总损失

$\mathcal{J}(\theta)=\frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i,y_i;\theta)\;+\;\frac{\lambda}{2}\sum_{l}\|W^{(l)}\|_F^2.$

### 2. AlexNet 的逐层尺寸与结构（典型实现）

#### 2.1 尺寸流（单图推理）

- 输入：$227\times227\times3$。

- Conv1（$11\times11$, $s=4$, $p=0$, $C_{out}=96$）→ $55\times55\times96$
ReLU → LRN → MaxPool（$3\times3$, $s=2$）→ $27\times27\times96$

- Conv2（$5\times5$, $s=1$, $p=2$, $C_{out}=256$）→ $27\times27\times256$
ReLU → LRN → MaxPool（$3\times3$, $s=2$）→ $13\times13\times256$

- Conv3（$3\times3$, $s=1$, $p=1$, $C_{out}=384$）→ $13\times13\times384$ → ReLU

- Conv4（$3\times3$, $s=1$, $p=1$, $C_{out}=384$）→ $13\times13\times384$ → ReLU

- Conv5（$3\times3$, $s=1$, $p=1$, $C_{out}=256$）→ $13\times13\times256$
ReLU → MaxPool（$3\times3$, $s=2$）→ $6\times6\times256$

- 展平 → $9216$
FC6：$9216\to4096$ → ReLU → Dropout
FC7：$4096\to4096$ → ReLU → Dropout
FC8：$4096\to1000$ → Softmax

### 3. 训练目标与优化算法（含动量与权重衰减）

#### 3.1 小批量 SGD（带动量）

- **梯度（对参数 **$\theta$**）：** $\nabla_\theta \mathcal{J}(\theta)$ 由反向传播得到。

- **动量更新：**

$v_{t}=\mu v_{t-1}+\nabla_\theta \mathcal{J}(\theta_t),\qquad\theta_{t+1}=\theta_t-\eta\big(\,v_{t}+\lambda\,\theta_t^{(\text{wd})}\big)$

- 其中 $\mu$ 为动量系数（如 $0.9$），$\eta$ 学习率，$\lambda$ 权重衰减；$\theta^{(\text{wd})}$ 通常只对权重项（不含偏置、归一化偏置等）施加。

#### 3.2 数据预处理与增强（影响分布与梯度方差）

- **均值消除：** 像素减去训练集 RGB 通道均值。

- **随机裁剪/镜像：** 训练时随机裁 $227\times227$（或近似）与水平翻转。

- **PCA 颜色扰动：** 令 RGB 像素数据的协方差矩阵的主成分为 $(p_1,p_2,p_3)$ 与特征值 $(\lambda_1,\lambda_2,\lambda_3)$，采样 $\alpha_i\sim\mathcal{N}(0,\sigma^2)$（如 $\sigma=0.1$），对每个像素做

$\Delta I = \sum_{i=1}^3 \alpha_i \lambda_i p_i,\qquad I\leftarrow I+\Delta I.$

### 4. 反向传播的核心推导（逐模块）

> 记号说明：对任一中间变量 $U$，其损失梯度记为 $\frac{\partial\mathcal{J}}{\partial U}$。链式法则用 $\odot$ 表示逐元素乘。

#### 4.1 Softmax \+ 交叉熵的梯度（关键结论与推导）

- logits：$s=W x + b$，概率 $p_k=\frac{e^{s_k}}{\sum_j e^{s_j}}$，损失 $\mathcal{L}=-\sum_k y_k\log p_k$。

- **关键梯度：**

$\frac{\partial \mathcal{L}}{\partial s_k}=p_k-y_k.$

- **推导要点：**

$\frac{\partial \mathcal{L}}{\partial s_k}= -\sum_j y_j \frac{1}{p_j}\frac{\partial p_j}{\partial s_k},\quad\frac{\partial p_j}{\partial s_k} = p_j(\delta_{jk}-p_k),$

- 整理即得 $p_k-y_k$。

- **进一步：** 对 FC8

$\frac{\partial \mathcal{L}}{\partial W}=(p-y)\,x^\top,\quad\frac{\partial \mathcal{L}}{\partial b}=p-y,\quad\frac{\partial \mathcal{L}}{\partial x}=W^\top (p-y).$

#### 4.2 全连接层（一般形式）

- 前向：$z=Wx+b$，后向给定 $\delta=\frac{\partial \mathcal{J}}{\partial z}$：

$\frac{\partial \mathcal{J}}{\partial W}=\delta\,x^\top,\quad\frac{\partial \mathcal{J}}{\partial b}=\delta,\quad\frac{\partial \mathcal{J}}{\partial x}=W^\top\delta.$

#### 4.3 ReLU 的梯度

- 前向：$y=\max(0,z)$；后向：

$\frac{\partial \mathcal{J}}{\partial z}=\frac{\partial \mathcal{J}}{\partial y}\odot \mathbf{1}_{z>0}.$

#### 4.4 Dropout 的梯度

- 训练时：$\tilde{x} = (m\odot x)/q$。后向：

$\frac{\partial \mathcal{J}}{\partial x}=\frac{m}{q}\odot \frac{\partial \mathcal{J}}{\partial \tilde{x}}.$

- 推理时无梯度分支（不使用掩码）。

#### 4.5 最大池化（MaxPool）梯度

- 前向保存每个池化窗口的**argmax** 位置 $(u^*, v^*)$。后向把梯度路由回该位置：

$\Big(\frac{\partial \mathcal{J}}{\partial X}\Big)_{i\cdot s+u^*,\,j\cdot s+v^*,\,c}\;+=\;\Big(\frac{\partial \mathcal{J}}{\partial Y}\Big)_{i,j,c},$

- 窗口内其他位置梯度为 0。

#### 4.6 卷积层的梯度（重点）

- 前向：$Z=W*X + b$。设后向来自上一层的梯度为 $\Delta=\frac{\partial \mathcal{J}}{\partial Z}$（尺寸 $H'\times W'\times C_{out}$）。

- **对偏置：** 每个输出通道 c 的梯度是该通道所有位置梯度之和：

$\frac{\partial \mathcal{J}}{\partial b_c}=\sum_{i,j}\Delta_{i,j,c}.$

- **对卷积核 **$W$**：** 是输入与 $\Delta$ 的“相关”（cross-correlation）：

$\frac{\partial \mathcal{J}}{\partial W_{u,v,d,c}}= \sum_{i,j} X_{i\cdot s+u-p,\; j\cdot s+v-p,\; d}\;\cdot\; \Delta_{i,j,c}.$

- **对输入 **$X$**：** 相当于把每个输出通道的 $\Delta_{:,:,c}$ 与**翻转**后的卷积核在通道维上做卷积并累加（考虑步幅/填充的反向映射）：

$\frac{\partial \mathcal{J}}{\partial X_{a,b,d}}= \sum_{c}\sum_{u,v}W_{u,v,d,c}\;\cdot\; \Delta_{\frac{a-p-u}{s},\,\frac{b-p-v}{s},\,c}$

- 其中仅当 $\frac{a-p-u}{s},\frac{b-p-v}{s}$ 为整数且落在 $[0,H'-1],[0,W'-1]$ 内时取值。实现中常用“上采样 \+ 全卷积/转置卷积”的等价计算或 **im2col** 矩阵化。

#### 4.7 LRN 的梯度（含分子与分母两路）

- 回忆：

$b^i = \frac{a^i}{D^{\beta}},\quad D = k + \frac{\alpha}{n}\sum_{j \in \mathcal{N}(i)} (a^j)^2$

- 设损失对 $b^i$ 的梯度为 $g^i=\frac{\partial \mathcal{J}}{\partial b^i}$，则对 $a^i$：

$\frac{\partial \mathcal{J}}{\partial a^i}= g^i\cdot D^{-\beta}\;+\;\sum_{m\in\mathcal{N}(i)} g^m\cdot a^m\cdot\Big(-\beta D^{-\beta-1}\Big)\cdot \frac{2\alpha}{n}\, a^i\cdot \mathbf{1}_{i\in\mathcal{N}(m)}.$

- 第一项来自分子 $a^i$；第二项来自分母 $D$ 对所有受影响通道的“反向侧抑制”。

### 5. 损失函数的批量形式与正则化项梯度

#### 5.1 批量交叉熵

- 批大小 $B$，样本 $(x^{(b)},y^{(b)})$：

$\frac{\lambda}{2}\sum_l |W^{(l)}|_F^2.$

#### 5.2 权重衰减（$L_2$）梯度

- 对任意权重矩阵 $W$：

$\frac{\partial}{\partial W}\Big(\frac{\lambda}{2}\|W\|_F^2\Big)=\lambda W.$

- 实际更新时常与动量/学习率一道合并到优化步骤中（见 3.1）。

### 6. AlexNet 的端到端训练/推理流程（无代码版“可执行思路”）

#### 6.1 初始化

1. 设定层级结构与超参（卷积核大小、步幅、填充、通道数、池化窗口、Dropout 概率等）。

2. 权重初始化：高斯/均匀分布的小方差随机数（可按 He/Xavier 机制：保证前向/反向方差稳定）。

3. 偏置初始化为零或小常数。

#### 6.2 数据通道

1. 读入图像，缩放到固定尺度（如最短边 256）。

2. 随机裁剪 $227\times227$、随机水平翻转。

3. 减去训练集均值；可加 PCA 颜色扰动。

#### 6.3 前向传播（单批次）

1. **Conv1 → ReLU → LRN → MaxPool**，记录池化 argmax、LRN 的中间量 $D$。

2. **Conv2 → ReLU → LRN → MaxPool**，同上记录必要中间量。

3. **Conv3 → ReLU → Conv4 → ReLU → Conv5 → ReLU → MaxPool**。

4. 展平 → **FC6 → ReLU → Dropout → FC7 → ReLU → Dropout → FC8**。

5. **Softmax** 得到 $p$；计算交叉熵损失（可加 $L_2$ 正则）。

#### 6.4 反向传播（单批次）

1. 从 **Softmax\+CE** 得到 $\delta^{(8)}=p-y$。

2. 依次对 **FC8、Dropout、ReLU、FC7、Dropout、ReLU、FC6** 回传：

    - 每层按 4.2/4.3/4.4 给出 $\frac{\partial \mathcal{J}}{\partial W}, \frac{\partial \mathcal{J}}{\partial b}, \frac{\partial \mathcal{J}}{\partial x}$。

3. 将梯度 reshape 成卷积张量形状，依次对 **Conv5→Conv4→Conv3→Conv2→Conv1** 回传：

    - ReLU：门控梯度（4.3）；

    - 池化：按 argmax 路由（4.5）；

    - LRN：用 4.7 的公式回传；

    - 卷积：用 4.6 计算对 $W,b,X$ 的梯度。

4. 在每层权重梯度上加上 $L_2$ 项 $\lambda W$（若在优化器中未合并）。

#### 6.5 参数更新

- 按 3.1 用 **SGD\+动量\+权重衰减** 更新全部可训练参数；必要时调整学习率（阶梯式或多项式衰减等）。

#### 6.6 推理（测试）

- 关闭 Dropout 与数据增强（仅做中心裁剪/尺度归一化）；前向一次得到概率向量 $p$，取 $\arg\max_k p_k$。

### 7. 关键细节的进一步理解与数学联系

#### 7.1 为什么 ReLU 有利于优化

Sigmoid 在 $|z|$ 大时 $\sigma'(z)\approx 0$，梯度消失；ReLU 的导数是 $\mathbf{1}_{z>0}$，在激活区间内保持常数，梯度路径更“直”。

#### 7.2 大卷积核 \+ 大步幅的感受野与下采样

Conv1 的 $11\times11, s=4$ 等价于早期强下采样，快速扩大感受野，降低后续计算量。尺寸公式确保输出为整数（$227\to55$）。

#### 7.3 LRN 的“侧抑制”数学效应

同一位置上通道间的二次项 $\sum (a^j)^2$ 在分母中抑制幅度较大的响应，使网络更关注“相对显著”的通道响应；反向中体现为通道间的耦合梯度（见 4.7 第二项）。

#### 7.4 Dropout 的期望保持与集成效果

训练相当于对指数多的“子网络”做参数共享；按 $1/q$ 缩放保证 $\mathbb{E}[\tilde{x}]=x$，推理等价于对大量子网络的近似平均。

### 8. 典型超参数与实践取值

- 批大小：如 128

- 学习率：如 $10^{-2}$ 起步，按验证集下降分段衰减

- 动量 $\mu$：0.9

- 权重衰减 $\lambda$：$5\times10^{-4}$

- Dropout：0.5（FC6、FC7）

- LRN：$k=2,\ \alpha=10^{-4},\ \beta=0.75,\ n=5$

### 9. 从零到一的“纸上实现顺序清单”

1. 明确每层尺寸与参数形状（用公式算出 $H',W'$）。

2. 写出前向：Conv（相关/卷积）、ReLU、LRN、Pool、FC、Softmax。

3. 写出后向：

    - Softmax\+CE：$\delta=p-y$。

    - FC：$dW=\delta x^\top,\ db=\delta,\ dx=W^\top\delta$。

    - ReLU：门控。

    - Dropout：乘同一批次的掩码并按 $1/q$ 缩放。

    - Pool：argmax 路由。

    - LRN：分子直传 \+ 分母耦合项。

    - Conv：对 $b$ 求和、对 $W$ 做相关、对 $X$ 做带翻转核的“全卷积”。

4. 组合梯度，做 **SGD\+动量\+权重衰减** 更新。

5. 用数据增强与学习率调度稳定训练，观察训练/验证曲线，按需调整。

### 10. 一眼看懂的核心公式（浓缩）

- 卷积输出尺寸：$\;H'=\big\lfloor\frac{H-k_h+2p}{s}\big\rfloor+1,\;W'=\big\lfloor\frac{W-k_w+2p}{s}\big\rfloor+1$

- Softmax\+CE 梯度：$\;\frac{\partial\mathcal{L}}{\partial s}=p-y$

- ReLU 梯度：$\;\frac{\partial\mathcal{J}}{\partial z}=\frac{\partial\mathcal{J}}{\partial y}\odot \mathbf{1}_{z>0}$

- MaxPool 梯度：路由至 argmax

- Conv 梯度（核）：$\;\frac{\partial \mathcal{J}}{\partial W_{u,v,d,c}}=\sum_{i,j} X_{i\cdot s+u-p,\; j\cdot s+v-p,\; d}\,\Delta_{i,j,c}$

- LRN：$\;b^i=\dfrac{a^i}{\big(k+\frac{\alpha}{n}\sum_{j\in\mathcal{N}(i)}(a^j)^2\big)^\beta}$

以上大篇幅的介绍，从**模块定义 → 尺寸计算 → 前后向梯度 → 优化与正则 → 端到端流程**系统化刻画了 AlexNet。按此逐步实现，便可完整复现其训练与推理机制。

## 完整例子

例子部分，我们使用Pytroch实现一个简易的AlexNet实现过程，包含训练、可视化和分析，帮助大家对于机制的理解~

```Python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

# 1. 数据准备
transform = transforms.Compose([
    transforms.Resize((224, 224)),                # 调整尺寸适配 AlexNet
    transforms.Grayscale(num_output_channels=3), # 转为 3 通道
    transforms.ToTensor()
])

# root 指向 MNIST 数据所在父目录 /MNIST
train_dataset = datasets.MNIST(root='./MNIST', 
                               train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./MNIST', 
                               train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义简化 AlexNet
class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleAlexNet().to(device)

# 3. 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练与测试
num_epochs = 3
train_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {train_losses[-1]:.4f} | Test Accuracy: {test_accuracy:.2f}%')

# 5. 可视化训练曲线
plt.figure(figsize=(10,5))
plt.plot(range(1,num_epochs+1), train_losses, marker='o', color='orange', label='Train Loss')
plt.plot(range(1,num_epochs+1), test_accuracies, marker='s', color='green', label='Test Accuracy')
plt.title('Training Loss & Test Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 6. 可视化预测结果
classes = [str(i) for i in range(10)]
model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(12,6))
for idx in range(8):
    plt.subplot(2,4,idx+1)
    img = images[idx].cpu().permute(1,2,0)
    plt.imshow(img)
    plt.title(f"GT: {classes[labels[idx]]}\nPred: {classes[preds[idx]]}", color='purple')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

我们这里，使用 **PyTorch** 实现了一个简化版的 **AlexNet**，用于对 **MNIST 手写数字数据集**进行分类。

1. **数据准备**

    - 将 MNIST 原始的 28×28 灰度图调整为 224×224 的 3 通道图像，以适配 AlexNet 输入。

    - 使用 `DataLoader` 按批加载训练集和测试集。

2. **模型构建**

    - 定义 `SimpleAlexNet` 类，包含：

        - **卷积特征提取层**：多层卷积 \+ ReLU \+ 最大池化

        - **全连接分类器**：线性层 \+ Dropout，用于将提取的特征映射到 10 类输出

    - 支持前向传播。

3. **训练设置**

    - 损失函数：交叉熵损失 (`CrossEntropyLoss`)

    - 优化器：Adam

4. **训练与测试**

    - 训练过程中记录每个 epoch 的训练损失

    - 测试集上计算分类准确率

    - 可视化训练损失与测试准确率的变化趋势

**结果可视化**

- 训练曲线（损失下降、准确率上升）

- 随机展示部分测试图像的预测结果，与真实标签对比

![21_神经网络-AlexNet-1.png](21_神经网络-AlexNet-1.png)

## 模型分析

### AlexNet 优缺点分析

**优点**：

1. **结构简单、易理解**：AlexNet 是较早的大型卷积神经网络，结构清晰，卷积 \+ ReLU \+ 池化 \+ 全连接，适合初学者学习卷积网络思想。

2. **强大的特征提取能力**：多层卷积可以提取低级到高级的图像特征，在复杂图像分类任务中表现良好。

3. **可迁移性好**：训练好的模型可以用于迁移学习，微调到其他图像分类任务。

4. **训练稳定**：使用 ReLU 和 Dropout 后收敛速度较快，减少过拟合。

**缺点**：

1. **参数量大**：AlexNet 的全连接层参数占比很高（尤其是 4096 维的全连接层），在小数据集或算力有限情况下容易过拟合。

2. **输入要求高**：要求输入图像较大（224x224），对于 MNIST 28x28 灰度图，需要先 resize 和通道扩展，带来额外计算和信息稀释。

3. **计算量大**：相比现代轻量级网络（如 MobileNet、ResNet-18），训练速度慢，推理效率低。

4. **结构过时**：缺乏批量归一化（BatchNorm）、残差连接，训练大规模网络时效果不如 ResNet 系列。

### AlexNet 与相似算法对比

|算法|参数量|特征提取能力|训练速度|推理速度|优势特点|劣势特点|
|---|---|---|---|---|---|---|
|AlexNet|高|高|中|中|易理解，经典卷积结构，可迁移学习|参数大，计算量大，过时|
|LeNet-5|低|低|高|高|适合小型数据集，如 MNIST|表达能力弱，复杂图像分类性能低|
|VGG-16|很高|高|低|低|更深层次特征提取，结构简单|参数过多，训练慢|
|ResNet-18|中|高|高|高|残差连接，训练深层网络稳定|架构复杂度较高，初学者理解困难|
|MobileNet|低|中|高|很高|轻量级，适合移动端和嵌入式|特征提取能力不如深层网络|

### 适用场景与算法选择

**AlexNet 优选情况**：

1. **中等规模图像分类任务**：数据量不是特别大，但希望用卷积网络学习图像特征。

2. **教育与学习**：理解卷积网络机制、卷积 \+ 池化 \+ 全连接的基本流程。

3. **迁移学习**：可以在更大数据集上训练，然后微调到类似任务。

**可以考虑其他算法的情况**：

1. **小型数据集**：如 MNIST、Fashion-MNIST，LeNet-5 或轻量级网络更高效，避免过拟合。

2. **大规模复杂图像**：ResNet、DenseNet 更深且训练稳定，特征表达更强。

3. **移动端/嵌入式场景**：MobileNet、ShuffleNet 等轻量级网络更适合，计算量小、推理快。

4. **追求最新性能**：现代架构（ResNet、EfficientNet、ViT）通常比 AlexNet 准确率更高。

总结来说，本例子使用 AlexNet 的主要价值在于 **学习卷积网络基础结构、训练流程和可视化特征**，在 MNIST 这样的简单数据集上可以工作，但如果追求效率或最新性能，可以考虑轻量级网络或深层残差网络。