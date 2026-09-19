# 对数变换

_面对右偏长尾时，我会先问“差异更像加法还是倍数”，再决定是否取对数。_

---

## 📋 为什么它能压缩长尾

对数把乘法关系变成加法关系：

\[
\log(ab)=\log(a)+\log(b)
\]

原始值从 10 增加到 100，与从 100 增加到 1000，都是扩大 10 倍；取对数后，两段距离相同。这对收入、交易额、文件大小、访问量等跨越多个数量级的特征很有用。

常见写法是：

\[
x'=\log(1+x)
\]

`log1p` 在 (x) 接近 0 时比直接计算 `log(1 + x)` 更稳定。[^1]

## 🎯 什么时候值得尝试

- 分布明显右偏，少数大值拉出很长的尾巴
- “翻倍”比“增加固定数值”更符合业务解释
- 残差方差随目标值增大，且变换后关系更接近线性

对数不是为了把每列都加工得更“好看”。如果零值有独立业务含义，或者正负号很重要，直接取对数可能改变问题本身。

## 🔧 代码先处理定义域

```python
import numpy as np

# 仅适用于 x >= 0 的计数或金额特征
X_train = X_train.copy()
X_test = X_test.copy()

X_train["order_count_log"] = np.log1p(X_train["order_count"])
X_test["order_count_log"] = np.log1p(X_test["order_count"])
```

不要为了让公式能算，随手用“全量数据最小值”决定平移常数。这既可能泄漏测试集信息，也让变换失去稳定的业务含义。

## ⚠️ 零值和负值怎么办

| 数据情况 | 我通常的处理 |
| --- | --- |
| (x\ge 0) | `np.log1p(x)` |
| (x>0) 且需要可调幂变换 | [Box-Cox](box-cox.md) |
| 同时包含正数、零和负数 | 考虑 Yeo-Johnson，而不是硬加常数 |
| 零表示“从未发生” | 额外保留是否为零的指示变量 |

scikit-learn 的 `PowerTransformer` 同时提供 Yeo-Johnson 和 Box-Cox；前者允许输入为负数。[^2]

## 🔍 如何判断有没有帮助

我不会只比较变换前后的直方图，还会看：

1. 交叉验证指标是否改善
2. 线性模型残差是否更稳定
3. 变换后的系数能否解释
4. 线上新数据是否仍满足定义域

## 📌 一句话记忆

当业务关系更接近“倍数变化”时，对数很自然；只因为分布偏斜就机械取对数，往往会制造新的解释问题。

## 🔗 参考资料

[^1]: NumPy developers. “numpy.log1p.” _NumPy Reference_. https://numpy.org/doc/stable/reference/generated/numpy.log1p.html

[^2]: scikit-learn developers. “PowerTransformer.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
