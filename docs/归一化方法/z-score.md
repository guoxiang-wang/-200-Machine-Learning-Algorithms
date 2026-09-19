# Z-score 标准化

_把每个值改写成“距离训练集均值多少个标准差”，是我最常用的数值特征基线。_

---

## 📋 它到底做了什么

\[
z = \frac{x-\mu}{\sigma}
\]

减去均值负责**中心化**，除以标准差负责**统一尺度**。在同一套统计口径下，(z=2) 表示这个值高于均值两个标准差。

> 📌 **容易误会的点：** Z-score 只保证训练数据按所用定义得到零均值和单位方差，并不会把偏态数据“变成正态分布”。

## 🎯 为什么常被当作默认方案

距离、点积和正则项都可能受数值尺度影响。特征一个以“元”为单位、另一个以“年”为单位时，未经处理的数值大小没有可比性。StandardScaler 不要求先知道上下界，也不会把未来样本硬压进固定区间。[^1]

我通常在这些模型前先试它：

- 逻辑回归、岭回归和 Lasso
- SVM、KNN 与 K-means
- 使用梯度优化、且输入尺度差异很大的模型

## 🔧 推荐写法

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(
    StandardScaler(),
    SVC(C=1.0, kernel="rbf"),
)
model.fit(X_train, y_train)
```

稀疏矩阵不能随意中心化，否则大量零值会变成非零值。遇到这种数据，应使用 `StandardScaler(with_mean=False)`，或者直接比较 [MaxAbs 缩放](max-abs.md)。[^1]

## 🔍 两个需要单独检查的问题

### 极端值

均值和标准差都会被极端值影响。某一列长尾很重时，我会同时画分位数或箱线图，再比较 `StandardScaler` 与 `RobustScaler`，而不是因为方法名字叫“标准化”就直接采用。

### 方差为零

常数列不能提供区分信息。scikit-learn 会保留这类列的缩放因子为 1，但建模前删除通常更清楚。[^1]

## 📌 一句话记忆

Z-score 统一的是中心和尺度，不是分布形状；它是一个好基线，但不是自动正确的答案。

## 🔗 参考资料

[^1]: scikit-learn developers. “StandardScaler.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
