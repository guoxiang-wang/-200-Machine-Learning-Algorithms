# Box-Cox 变换

_不预先认定该开方还是取对数，而是让训练数据帮助选择幂参数。_

---

## 📋 公式

对严格为正的 (x)，Box-Cox 变换定义为：

\[
y(\lambda)=
\begin{cases}
\dfrac{x^\lambda-1}{\lambda}, & \lambda\ne 0 \\
\log(x), & \lambda=0
\end{cases}
\]

​(lambda=1) 接近线性变换，​(lambda=0) 对应对数，​(lambda=0.5) 与平方根有关。SciPy 可以通过最大似然估计 ​(lambda)。[^1]

## 🎯 我为什么会用它

- 特征或回归目标严格为正，而且右偏明显
- 希望用统一方法在一组幂变换中选择参数
- 线性模型的方差稳定性或线性关系需要改善

它不是“让任何数据自动正态”的按钮。最优 ​(lambda) 取决于训练样本，换一批数据可能会变化。

## 🔧 放进流水线

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge

model = make_pipeline(
    PowerTransformer(method="box-cox", standardize=True),
    Ridge(alpha=1.0),
)
model.fit(X_train_positive, y_train)
```

`PowerTransformer` 会估计每个特征的参数，并可在幂变换后继续做零均值、单位方差标准化。Box-Cox 要求输入严格为正；Yeo-Johnson 则允许正数和负数。[^2]

## ⚠️ 三个常见问题

### 零值不能直接输入

“统一加 1”只在 1 有明确单位意义时比较自然。若测量单位变化，加 1 的影响也会变化。先确认零的含义，再决定平移、换用 Yeo-Johnson，还是选别的方法。

### 参数也会泄漏

​(lambda) 必须只从训练集估计。把全部数据交给 `boxcox` 后再拆分，同样属于数据泄漏。

### 目标变换要记得逆变换

对回归目标做 Box-Cox 后，模型误差发生在变换尺度上。最终报告指标时应回到原单位，并说明反变换后的偏差。

## 🔍 与手工选择的关系

数据与业务已经明确支持对数时，直接用 `log1p` 更容易解释。Box-Cox 更适合“知道需要幂变换，但不确定指数”的情形。

## 📌 一句话记忆

Box-Cox 的价值是从训练数据估计幂参数；它的硬条件是数据严格为正，而且参数估计必须遵守训练/验证边界。

## 🔗 参考资料

[^1]: SciPy community. “scipy.stats.boxcox.” _SciPy API Reference_. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html

[^2]: scikit-learn developers. “PowerTransformer.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
