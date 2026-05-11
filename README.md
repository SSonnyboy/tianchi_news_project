# 天池新闻推荐竞赛 - 推荐系统学习项目

## 项目简介

基于天池新闻推荐竞赛数据，实现完整的召回-排序推荐系统 pipeline。召回阶段采用 4 路融合（ItemCF、Swing、YouTubeDNN、冷启动），排序阶段以 LightGBM LambdaRanker 为基线，并讨论 DCN、DeepFM、AutoTINT、RankMixer 等特征交叉模型。

## 项目结构

```
├── src/                        # 共享基础设施
│   ├── config.py               # 路径与超参数配置
│   ├── data_loader.py          # 数据加载、LOO 划分、字典构建
│   ├── metrics.py              # Recall@K, MRR@K, AUC, NDCG@K
│   ├── features.py             # 排序特征工程 (19 维)
│   ├── negative_sampling.py    # 负采样策略
│   ├── recall/                 # 召回算法
│   │   ├── itemcf.py           # ItemCF (加权变体)
│   │   ├── swing.py            # Swing 算法
│   │   ├── youtube_dnn.py      # YouTubeDNN 双塔模型
│   │   ├── hot.py              # 热门召回
│   │   ├── i2i_30s.py          # 30 秒间隔 i2i
│   │   └── combiner.py         # 多路召回融合
│   └── ranking/                # 排序模型
│       ├── lgb_ranker.py       # LightGBM Ranker (LambdaRank)
│       └── base.py             # 排序模型基类
├── solutions/                  # 方案流水线
│   ├── base_pipeline.py        # 基类 (load -> recall -> ranking -> evaluate)
│   └── solution_unified.py     # 统一方案: 4 路召回 + LGB Ranker
├── docs/                       # 技术文档
│   ├── feature_crossing.md     # 特征交叉模型讨论 (DCN/DeepFM/AutoTINT/RankMixer)
│   ├── data_process.md         # 数据处理说明
│   └── metric.md               # 评估指标说明
├── notebooks/                  # 实验 notebook
├── data/                       # 原始数据 (需自行下载)
└── outputs/                    # 缓存与结果
```

## 召回架构

4 路召回融合，每路独立打分后加权合并：

| 通道 | 算法 | 权重 | 说明 |
|------|------|------|------|
| ItemCF | 加权 ItemCF | 1.0 | 基于点击共现 + 位置衰减 + 创建时间相似度 |
| Swing | Swing 算法 | 1.0 | 基于用户对共点击物品的交互模式 |
| YouTubeDNN | 双塔 DNN | 1.2 | user tower (embedding + mean pooling + DNN) vs item tower |
| 冷启动 | Hot + 30s-i2i | 0.8 | 热门召回 + 30 秒间隔 i2i 合并，覆盖新用户/新物品 |

融合方式：per-user MinMax 归一化 -> 加权求和 -> Top-50

## 排序架构

### 基线：LightGBM LambdaRanker

- **损失函数**: LambdaRank（直接优化 NDCG）
- **特征**: 19 维（用户统计、物品统计、时间差、文本差异、上下文、交叉、召回分数）
- **验证**: 80/20 用户划分，GroupKFold

### 特征交叉模型探索方向

详见 `docs/feature_crossing.md`：

| 模型 | 核心思路 | 优势 |
|------|---------|------|
| DCN-V2 | Cross Network 显式有限阶交叉 | 参数效率高，工业界广泛使用 |
| DeepFM | FM 二阶 + DNN 高阶并行 | 无需手工特征工程 |
| AutoTINT | Self-Attention 自动发现交互 | 可解释性强 |
| RankMixer | MLP-Mixer token-mixing | 高效，适合特征数多的场景 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

将天池竞赛数据放入 `data/` 目录：

- `train_click_log.csv` - 训练集点击日志 (111 万条，20 万用户)
- `articles.csv` - 文章元信息 (36 万篇)
- `articles_emb.csv` - 文章 embedding (250 维)

### 3. 运行

```python
from solutions.solution_unified import SolutionUnified
pipeline = SolutionUnified()
metrics = pipeline.run()
```

## 数据划分

采用 **Leave-One-Out** 划分：按时间排序后，每个用户的最后一次点击作为测试标签，其余作为训练数据。

## 评估指标

| 阶段 | 指标 | 含义 |
|------|------|------|
| 召回 | Recall@K | 前 K 个召回结果中包含正确答案的用户比例 |
| 召回 | MRR@5 | 平均倒数排名 (前 5) |
| 排序 | AUC | ROC 曲线下面积 |
| 排序 | NDCG@5 | 归一化折损累积增益 (前 5) |
