# 平方根变换

_它也是压缩右尾的方法，但力度比对数温和，尤其适合从零开始的计数。_

---

## 📋 变换方式

对非负数据：

\[
x'=\sqrt{x}
\]

原始值 (0,1,4,9,16) 会变成 (0,1,2,3,4)。大值之间的距离被压缩，小值附近仍保留较多区分度。

平方根变换常出现在计数数据的探索阶段。它并不保证正态性，也不替代适合计数分布的模型。

## 🎯 为什么有时比对数合适

- 数据包含大量合法的 0，不想人为加 1 后再解释
- 右偏存在，但没有跨越很多数量级
- 希望保留比对数更多的大值差异

可以把压缩强度粗略理解为：原尺度 < 平方根 < 对数。这个顺序只是直觉，最终仍要以验证结果和业务解释为准。

## 🔧 简单实现

```python
import numpy as np

if (X_train["incident_count"] < 0).any():
    raise ValueError("incident_count 必须是非负计数")

X_train = X_train.assign(
    incident_count_sqrt=np.sqrt(X_train["incident_count"])
)
X_test = X_test.assign(
    incident_count_sqrt=np.sqrt(X_test["incident_count"])
)
```

如果变换逻辑要进入交叉验证流水线，可以使用 `FunctionTransformer(np.sqrt)`；但定义域检查仍应保留。[^1]

## ⚠️ 关于负数

有时会看到“带符号平方根”：

\[
x'=\operatorname{sign}(x)\sqrt{|x|}
\]

它在数学上可计算，却未必有清楚的业务意义。遇到正负都有的特征，我更愿意先理解负值代表什么，再决定保留原尺度、使用 Yeo-Johnson，还是拆分符号与绝对值。

## 🔍 检查清单

- 训练集和未来数据是否都不会出现非法负值
- 0 是真实观测，还是缺失值的占位符
- 变换后模型指标是否改善
- 预测结果回到原尺度后是否仍易解释

## 📌 一句话记忆

平方根是处理非负计数长尾的一种温和选择；它解决尺度问题，不替代对数据生成过程的理解。

## 🔗 参考资料

[^1]: scikit-learn developers. “FunctionTransformer.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html
