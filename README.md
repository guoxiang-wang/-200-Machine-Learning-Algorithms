# 机器学习笔记

_把学习过程中反复查过、算过和踩过坑的内容整理成可检索的中文笔记。_

---

[在线阅读](https://guoxiang-wang.github.io/-200-Machine-Learning-Algorithms/)

## 📚 目前整理到哪里

| 主题 | 内容 | 状态 |
| --- | --- | --- |
| 特征缩放与分布变换 | Min-Max、Z-score、稳健缩放、幂变换等 | 已重新整理 |
| 参数估计 | 最小二乘、最大似然、贝叶斯估计、EM | 持续校对 |
| 异常值处理 | 识别、截断、替换和分箱 | 持续校对 |
| 神经网络 | 基础结构与经典网络 | 持续校对 |

这不是算法百科。我更在意一个方法为什么要用、什么情况下会失效，以及代码里最容易忽略的边界条件。

## 🔍 推荐入口

- 不确定数值特征该怎么处理：从[特征缩放与分布变换](docs/归一化方法/index.md)开始
- 想比较不同估计思路：查看[参数估计方法](docs/参数估计方法/index.md)
- 数据里出现异常点：查看[异常值处理方法](docs/异常值处理方法/index.md)
- 复习网络结构：查看[神经网络](docs/神经网络/index.md)

## 🗂️ 仓库结构

```text
.
├── docs/                 # 公开发布的笔记
├── tools/                # 本地构建检查
├── .github/workflows/    # GitHub Pages 部署
├── mkdocs.yml            # 站点配置与导航
└── requirements.txt      # 最小构建依赖
```

原始 Obsidian 笔记不进入仓库；`docs/` 中的内容会经过人工合并、校对后再发布。这样做是为了让公开版本保持清楚，而不是把本地资料原样堆上来。

## 🔧 本地预览

```powershell
python -m pip install -r requirements.txt
python -m mkdocs serve
```

提交前可以运行：

```powershell
.\tools\check-site.ps1
```

推送到 `main` 后，GitHub Actions 会重新构建并发布站点。
