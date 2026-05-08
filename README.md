# 天池新闻推荐竞赛 - 统一学习代码库

## 项目简介

本项目将天池新闻推荐竞赛的 4 份获奖方案整合为一个统一的代码库，用于学习推荐系统的召回与排序流程。

| 方案 | 召回策略 | 排序模型 |
|------|---------|---------|
| **S1** | ItemCF (加权) + YouTubeDNN | 无 (纯召回) |
| **S2** | ItemCF + Bi-network + Word2Vec | LightGBM Classifier |
| **S3** | 热门召回 + 30秒间隔 i2i | LightGBM Ranker |
| **S5** | ItemCF + Swing + Item2Vec | LightGBM Ranker |

所有方案共享相同的数据划分（Leave-One-Out）和评估指标，可直接横向对比。

## 项目结构

```
├── src/                        # 共享基础设施
│   ├── config.py               # 路径与超参数配置
│   ├── data_loader.py          # 数据加载、LOO 划分、字典构建
│   ├── metrics.py              # Recall@K, MRR@K, AUC, NDCG@K
│   ├── features.py             # 排序特征工程
│   ├── negative_sampling.py    # 负采样策略
│   ├── recall/                 # 8 种召回算法
│   │   ├── itemcf.py           # ItemCF (3 种变体)
│   │   ├── binetwork.py        # Bi-network
│   │   ├── swing.py            # Swing 算法
│   │   ├── i2i_30s.py          # 30 秒间隔 i2i
│   │   ├── hot.py              # 热门召回
│   │   ├── word2vec_recall.py  # Word2Vec + Annoy
│   │   ├── item2vec_recall.py  # Item2Vec + Faiss
│   │   ├── youtube_dnn.py      # YouTubeDNN 双塔模型
│   │   └── combiner.py         # 多路召回融合
│   └── ranking/                # 排序模型
│       ├── lgb_classifier.py   # LightGBM Classifier
│       └── lgb_ranker.py       # LightGBM Ranker (LambdaRank)
├── solutions/                  # 4 个方案流水线
├── notebooks/                  # 对比实验 notebook
├── data/                       # 原始数据 (需自行下载)
└── outputs/                    # 缓存与结果
```

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

### 3. 运行对比实验

```bash
cd notebooks
jupyter notebook 01_run_all_solutions.ipynb
```

或在 Python 中直接调用：

```python
from solutions.solution1_news_recommender import Solution1NewsRecommender
s1 = Solution1NewsRecommender()
metrics = s1.run()  # 加载 → 召回 → 评估
```

## 数据划分

采用 **Leave-One-Out** 划分：按时间排序后，每个用户的最后一次点击作为测试标签，其余作为训练数据。这是学术推荐系统论文中最标准的评估协议。

## 评估指标

| 阶段 | 指标 | 含义 |
|------|------|------|
| 召回 | Recall@K | 前 K 个召回结果中包含正确答案的用户比例 |
| 召回 | MRR@5 | 平均倒数排名 (前 5) |
| 排序 | AUC | ROC 曲线下面积 |
| 排序 | NDCG@5 | 归一化折损累积增益 (前 5) |

## 核心算法速查

### 召回算法

| 算法 | 来源 | 核心公式 |
|------|------|---------|
| ItemCF | S1/S2/S5 | `sim[i][j] += loc_weight / log(len+1)`, 归一化 `sqrt(cnt_i * cnt_j)` |
| Bi-network | S2 | `sim[i][j] += 1 / (log(users_i+1) * log(items_u+1))` |
| Swing | S5 | `sim[i][j] += 1 / (alpha + \|co_users\|)`, alpha=5.0 |
| 30s-i2i | S3 | 统计恰好间隔 30 秒点击的文章对共现次数 |
| YouTubeDNN | S1 | 双塔: user_emb = DNN(user_emb \|\| mean(hist_emb)), dot(item_emb) |

### 排序模型

| 模型 | 方案 | 损失函数 |
|------|------|---------|
| LGBClassifier | S2 | Binary Cross-Entropy (pointwise) |
| LGBRanker | S3/S5 | LambdaRank (listwise, 直接优化 NDCG) |

### 负采样策略

| 策略 | 方案 | 说明 |
|------|------|------|
| 召回即负样本 | S2 | 召回候选中非 ground-truth 的作为负样本 |
| 双轴采样 | S3/S5 | 按用户 + 按物品两个维度各采样 1-5 个负样本 |
