# 二值化

_把连续数值变成“是否越过阈值”；它适合明确的业务判断，但会主动丢掉大量信息。_

---

## 📋 数学定义

设阈值为 (t)：

\[
x'=
\begin{cases}
1, & x>t \\
0, & x\le t
\end{cases}
\]

scikit-learn 的 `Binarizer` 默认使用“大于阈值”为 1，因此刚好等于阈值的值会得到 0。[^1] 这个边界细节必须和业务口径一致。

## 🎯 什么情况下值得丢掉数值

- 阈值本身有清晰含义，例如“是否逾期”“是否超过检测下限”
- 原始测量在阈值两侧可靠，但精确数值噪声很大
- 模型需要一个额外的状态特征，同时仍保留原连续列

很多时候，我会新增 `is_overdue`，而不是用它覆盖 `days_overdue`。这样模型既能看到是否越界，也保留超过了多少。

## 🔧 两种写法

业务规则明确时，直接写判断最清楚：

```python
df["is_adult"] = (df["age"] >= 18).astype("int8")
```

用于流水线、并且边界规则与 scikit-learn 一致时：

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0)
X_binary = binarizer.transform(X)
```

注意：上面的实现是 (x>0)，不是 (x\ge 0)。

## ⚠️ 阈值从哪里来

| 阈值来源 | 做法 |
| --- | --- |
| **法规或业务定义** | 固定阈值，并在代码和文档中写清边界 |
| **测量设备限制** | 使用检测下限，同时保留缺失/未检出语义 |
| **通过模型效果选择** | 只在训练集和验证集上调参，测试集最后使用 |
| **看完整数据后拍脑袋** | 避免，这会引入泄漏和不可复现性 |

## 🔍 它不是普通归一化

缩放通常保留排序和大部分相对距离；二值化只保留阈值两侧的状态。把 61 分和 99 分都变成 1 后，两者差异已经无法恢复。因此它应该由任务含义驱动，而不是为了“统一范围”。

## 📌 一句话记忆

阈值有意义时，二值化能让特征很清楚；阈值没有依据时，它只是把信息压扁。

## 🔗 参考资料

[^1]: scikit-learn developers. “Binarizer.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
