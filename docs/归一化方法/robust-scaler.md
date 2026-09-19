# 稳健缩放

_原笔记把它叫“分位数缩放”；更准确地说，这一页讨论的是以中位数和四分位距为基准的 RobustScaler。_

---

## 📋 为什么不用均值和标准差

对包含极端值的特征，均值和标准差可能被少数样本拖动。稳健缩放改用中位数 (Q_2) 与四分位距 (IQR=Q_3-Q_1)：

\[
x' = \frac{x-Q_2}{Q_3-Q_1}
\]

默认设置下，中位数会映射到 0，中间 50% 数据的跨度成为主要尺度。它不会删除离群点，也不会保证所有值落在 ([-1,1])。[^1]

## 🎯 我会在这些时候试它

- 金额、时长、访问次数等特征有明显长尾
- 极端值是真实业务记录，不能简单删除
- 线性模型、SVM 或距离模型又确实需要统一尺度

先确认异常值的来源仍然很重要。录入错误应该修数据，不应该交给缩放器“遮住”。

## 🔧 代码与参数

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

model = make_pipeline(
    RobustScaler(quantile_range=(25.0, 75.0)),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
```

`quantile_range` 不一定非要用 25% 到 75%。范围放宽会使用更多样本，缩放更接近整体；范围收窄会更关注中间区域。参数仍然只能从训练集估计。[^1]

## 🔍 与 QuantileTransformer 的区别

两者都使用分位数，但目标不同：

| 方法 | 主要动作 | 是否明显改变分布形状 |
| --- | --- | --- |
| **RobustScaler** | 减中位数、除以分位数范围 | 通常不会 |
| **QuantileTransformer** | 按经验累计分布映射到均匀或正态分布 | 会 |

因此把 RobustScaler 简称为“分位数缩放”尚可理解，但不要把它和分位数变换当成同一个算法。[^2]

## ⚠️ 代价

稳健不等于永远更好。样本很少时，四分位数本身可能不稳定；数据本来接近对称且没有明显极端值时，StandardScaler 往往更容易解释。

## 📌 一句话记忆

当极端值真实存在、又不希望它们决定整列尺度时，使用中位数和 IQR。

## 🔗 参考资料

[^1]: scikit-learn developers. “RobustScaler.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

[^2]: scikit-learn developers. “Compare the Effect of Different Scalers on Data with Outliers.” _scikit-learn Examples_. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
