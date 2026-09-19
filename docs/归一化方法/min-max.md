# Min-Max 缩放

_把训练集中的最小值映射到下界、最大值映射到上界；公式简单，边界条件却值得认真处理。_

---

## 📋 公式和直觉

对一个特征 (x)，映射到 ([0,1]) 的公式是：

\[
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

如果目标区间是 ([a,b])，再做一次线性映射：

\[
x'' = a + x'(b-a)
\]

例如训练数据为 (10,20,30)，缩放后是 (0,0.5,1)。它保留了样本之间的相对间距，没有改变排序。

## 🎯 我在什么情况下使用

- 输入必须落入已知范围，例如某些神经网络输入或需要组合多个评分
- 特征边界稳定，而且极端值本身不是录入错误
- 希望变换后的数值仍然容易直观解释

如果训练集存在一个特别大的值，其余样本会被挤在很窄的区间里。Min-Max 缩放不会削弱离群点，只是把所有值一起重新标尺。[^1]

## 🔧 一段够用的代码

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

numeric_columns = ["age", "monthly_income"]

preprocess = ColumnTransformer(
    [("numeric", MinMaxScaler(), numeric_columns)],
    remainder="passthrough",
)

model = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
```

流水线的价值不在于少写两行代码，而在于交叉验证时不会提前看到验证折的最小值和最大值。

## ⚠️ 实际项目里的边界

### 新数据可以越界

测试样本如果大于训练集最大值，变换结果会大于 1；小于训练集最小值则会小于 0。`MinMaxScaler(clip=True)` 可以截断，但截断会丢掉“超出多少”的信息。[^1]

### 常数列没有缩放意义

手写公式时，(x_{\max}=x_{\min}) 会造成除零。与其补一个很小的常数，我更倾向于先删除没有变化的特征。

### 别在拆分数据前缩放

错误顺序是“全量缩放 → 划分训练集和测试集”。正确顺序是先拆分，再让缩放器只 `fit` 训练集。

## 📌 一句话记忆

需要固定区间、边界又比较稳定时用 Min-Max；只要极端值开始主导最大最小值，就应该比较 [稳健缩放](robust-scaler.md)。

## 🔗 参考资料

[^1]: scikit-learn developers. “MinMaxScaler.” _scikit-learn API Reference_. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
