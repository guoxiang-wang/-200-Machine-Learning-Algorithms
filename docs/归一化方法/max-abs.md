# MaxAbs 缩放

_用每列的最大绝对值做除数，不移动零点，因此能保留稀疏矩阵中的零。_

---

## 📋 计算方式

\[
x' = \frac{x}{\max(|x|)}
\]

训练集中的结果通常落在 ([-1,1]) 内。全为非负数时，范围是 ([0,1])。与 Min-Max 不同，它不会减去最小值，所以原来的 0 仍然是 0。

## 🎯 它真正擅长的场景

我主要在稀疏特征上考虑 MaxAbs，例如词频、计数向量或 one-hot 后仍想做简单缩放的矩阵。因为不做中心化，矩阵不容易突然变得稠密。scikit-learn 的实现也明确支持稀疏 CSR/CSC 输入。[^1]

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    MaxAbsScaler(),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train_sparse, y_train)
```

## ⚠️ 它没有解决的问题

- 最大绝对值仍然会被单个极端样本控制
- 它不负责中心化，也不改变偏态分布
- 测试值超出训练集范围时，结果仍可能超过 ([-1,1])

所以“保留稀疏性”是选择它的主要理由，而不是“它比其他缩放器更稳健”。如果数据不稀疏且极端值明显，我会优先比较 [稳健缩放](robust-scaler.md)。

## 🔍 一个手算例子

对 (x=[-4,0,2])，最大绝对值是 4：

\[
x'=[-1,0,0.5]
\]

符号、零点和相对比例都保留下来，这正是它用于稀疏数据时的直觉。

## 📌 一句话记忆

MaxAbs 是“为了不破坏零而缩放”，不是处理离群点的工具。

## 🔗 参考资料

[^1]: scikit-learn developers. “MaxAbsScaler.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
