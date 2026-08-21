---
title: "Z-Score 标准化"
description: "异常值处理方法 主题下的 Z-Score 标准化 条目，已重写整理为适合公开分享的版本。"
---

> 这一页重新梳理 Z-Score 标准化 的判断规则和处理思路，尽量让读者先看懂“为什么”，再看“怎么做”。

## 快速理解
- 先判断点位是不是明显偏离主体分布。
- 再决定是删除、替换、截断还是分组处理。
- 处理后最好再检查分布是否更稳。

## 完整例子

我们选择一个**商品定价分析**的场景。当一家电商平台的数据分析师，收到一个数据集，记录了某一类产品的价格。你怀疑其中有些价格被误填（过高或过低），需要先排除这些异常值再做后续分析，例如定价策略、折扣策略等。

因此，第一步是要进行**异常值检测**，这时候我们选择使用 **Z-Score 标准化方法**。

数据集特征：

- 商品价格（单位：美元）

- 商品销量

- 类别标签（可选）

我们重点关注「价格」这一数值型变量，应用 Z-Score 对其进行标准化，识别出不合理的价格。

### 代码实现

```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 构造商品价格数据
np.random.seed(42)
normal_prices = np.random.normal(loc=100, scale=15, size=300)
outliers_high = np.random.uniform(200, 300, 5)
outliers_low = np.random.uniform(20, 40, 3)
all_prices = np.concatenate([normal_prices, outliers_high, outliers_low])

df = pd.DataFrame({'price': all_prices})

# 2. 计算 Z 值
mean_price = df['price'].mean()
std_price = df['price'].std()
df['z_score'] = (df['price'] - mean_price) / std_price

# 3. 标注异常值
df['is_outlier'] = df['z_score'].abs() > 3

# 4. 可视化：价格分布 + 异常点高亮
plt.figure(figsize=(14, 6))
sns.histplot(df['price'], bins=40, kde=True, color="#33C1FF", label="价格分布")
plt.axvline(mean_price, color='black', linestyle='--', label='均值')
plt.axvline(mean_price + 3*std_price, color='red', linestyle='--', label='+3σ')
plt.axvline(mean_price - 3*std_price, color='red', linestyle='--', label='-3σ')
plt.title("商品价格分布与Z-Score异常值")
plt.xlabel("价格（美元）")
plt.ylabel("频率")
plt.legend()
plt.show()

# 5. 可视化：Z 值分布图
plt.figure(figsize=(14, 6))
sns.scatterplot(x=range(len(df)), y=df['z_score'], hue=df['is_outlier'], palette=["green", "red"], s=60)
plt.axhline(3, color='red', linestyle='--')
plt.axhline(-3, color='red', linestyle='--')
plt.title("Z-Score 分布图：异常值高亮")
plt.xlabel("样本索引")
plt.ylabel("Z-Score")
plt.show()

# 6. 打印异常值
print("\n检测到的异常值如下：")
print(df[df['is_outlier']])
```

**步骤1：生成数据**

```Python
normal_prices = np.random.normal(loc=100, scale=15, size=300)
```

- 生成 300 个价格，平均值为 100，标准差为 15，模拟「正常价格分布」

- `loc` 是均值，`scale` 是标准差

```Python
outliers_high = np.random.uniform(200, 300, 5)
outliers_low = np.random.uniform(20, 40, 3)
```

- 人工添加 8 个明显偏离的数据（高得离谱，低得不正常）

```Python
all_prices = np.concatenate([normal_prices, outliers_high, outliers_low])
```

- 将正常值和异常值合并，构成我们要分析的价格数据

**步骤2：Z-Score 标准化**

```Python
mean_price = df['price'].mean()
std_price = df['price'].std()
df['z_score'] = (df['price'] - mean_price) / std_price
```

这是核心步骤，计算每个价格的 Z 值。Z 值大于 3 或小于 -3 被视为异常值。

**步骤3：判断异常值**

```Python
df['is_outlier'] = df['z_score'].abs() > 3
```

用绝对值判断，Z 值大于 3 的视为「异常点」，用布尔变量记录。

**步骤4：价格分布可视化**

```Python
sns.histplot(..., kde=True)
```

- 绘制直方图 \+ 核密度曲线，显示整体分布形态

- 用红色虚线标出 ±3σ 区间外，即 Z 值超过 3 的界限

![20_异常值处理方法-Z-Score 标准化-1.png](图片和附件/20_异常值处理方法-Z-Score%20标准化-1.png)

**步骤5：Z 值分布图**

```Python
sns.scatterplot(..., hue=df['is_outlier'])
```

- 横坐标是索引（为了展示点的位置），纵坐标是 Z 值

- 使用红/绿两色区分异常点与正常点，一目了然

![20_异常值处理方法-Z-Score 标准化-2.png](图片和附件/20_异常值处理方法-Z-Score%20标准化-2.png)

**步骤6：输出异常值**

```Python
print(df[df['is_outlier']])
```

展示被标记为异常的记录，可以进一步删除、修正或标注。

### 分析与解释：哪些是异常值、为什么？

在我们的例子中：

- 正常价格大多集中在 85~115 之间

- 添加的异常值如 250、275、30、35 等，Z 值明显超过 ±3

- 这些值不是因为市场规律而产生的，而是：

    - 人为误输

    - 爬虫错误

    - 数据单位错写

这些都属于应清洗的数据。

图像直观地显示了哪些点离群，红色的点在 Z 值图中高高在上或远低于下界。

### Z-Score 标准化的优点与缺点

**优点**：

1. **简洁直观**：只需知道平均值和标准差，就能判断一个点是否异常。

2. **标准化能力强**：不同数据集之间可以通用比较，因为都是以「标准差单位」度量。

3. **可扩展性好**：可以快速应用到多维度上，例如多变量联合判断。

4. **与正态分布模型契合度高**：如果数据服从正态分布，Z-Score 判断异常是非常准确的。

**缺点**：

1. **对分布形状敏感**：如果数据不是正态分布，Z 值的判断可能不准（例如偏态分布会误判正常值为异常）。

2. **受极端值影响大**：极端值本身会影响均值和标准差，使得标准化的效果变差，甚至掩盖其他异常值。

3. **不适用于分类变量或离散变量**：只能用于连续型数值数据。

4. **静态阈值限制**：阈值 ±3 只是经验值，某些业务场景可能需更灵活判断。

### Z-Score 方法的适用场景与限制

**适合的情况**：

- 数据服从或近似服从**正态分布**

- 对数值型数据，尤其是单一维度的分析

- 数据量足够大时，统计特性较稳定

- 异常值显著偏离均值，不是渐变异常（而是跳变）

**不适合的情况**：

- 数据为**严重偏态分布**（如收入分布、销售分布）

- 有很多重复值或离散型数据（如1、2、3评分）

- 异常值不明显或逐渐变化（如行为欺诈、渐进攻击）

- 数据中本身含噪或有明显季节性

## 总结

Z-Score 方法的**最大优势在于其直观性和简单性**，可以快速对数据做初步清洗，适合做前期的数据探索（EDA）或异常值识别任务。尤其在电商、金融、传感器信号处理、工业质量控制等领域，都是极常用的工具。

但需要注意的是，Z-Score 方法的适用前提是**数据的标准差和均值可以客观反映中心趋势和离散程度**。在严重偏态或异常点占比很高的情况下，Z 值容易失真。

因此，在实际使用中，我们往往会：

- 先用可视化判断数据分布形态

- 结合业务知识设定合理的 Z 值阈值（如 ±2.5）

- 与其他方法（如 IQR、MAD）联合使用，提高鲁棒性

例如：**IQR方法（四分位数）** 或 **Isolation Forest**。