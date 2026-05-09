# 学习路线：面向简历面试的推荐系统代码库

## 概述

本指南帮助你从零开始啃下这份推荐系统代码库，按"由浅入深、先跑通再理解"的原则组织。每一步都对应面试高频考点。

建议总时长：10 天（每天 2-3 小时）。

---

## 阶段 1: 跑通全链路 (1-2 天)

**目标**：先跑起来，建立直觉。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 把数据文件放到 data/ 目录
#    - train_click_log.csv
#    - articles.csv
#    - articles_emb.csv

# 3. 运行最简单的方案
python -c "
from solutions.solution1_news_recommender import Solution1NewsRecommender
s1 = Solution1NewsRecommender()
metrics = s1.run()
"
```

观察输出的 Recall@5、MRR@5 是多少，不需要理解代码细节。

**面试价值**：能说清"我跑通了一个推荐系统的召回+评估流程"。

---

## 阶段 2: 理解数据流 (1 天)

**核心文件**：`src/data_loader.py`

逐函数阅读，搞懂每一步在做什么：

| 函数 | 面试知识点 |
|------|-----------|
| `load_raw_data()` | 数据读取、dtype 优化、去重 |
| `leave_one_out_split()` | **Leave-One-Out 评估协议** — 面试必问 |
| `build_user_item_time_dict()` | 用户行为序列的组织方式 |
| `build_item_info_dicts()` | 物品属性字典的构建 |

**自测问题**：

- [ ] 为什么要用 Leave-One-Out 而不是随机 split？
- [ ] 单点击用户怎么处理？为什么？
- [ ] `{user_id: [(item_id, timestamp), ...]}` 这个数据结构有什么好处？

**面试标准答案**：

1. Leave-One-Out 模拟"预测用户下一次点击"的真实场景，避免数据泄露。随机 split 会导致同一用户的点击同时出现在训练和验证集中。
2. 单点击用户的那条点击同时放入训练集和测试集（数据泄露的妥协），否则该用户无法被评估。
3. 按时间排序的 (item, timestamp) 元组列表天然支持序列建模、时间窗口过滤、位置权重计算。

---

## 阶段 3: 攻克召回算法 (3-5 天)

**核心文件**：`src/recall/` 目录

按难度递进学习：

### 第 1 步：ItemCF (`src/recall/itemcf.py`)

**学习要点**：

1. 先看 `variant="baseline"` — 最简单的共现计数
2. 再看 `variant="weighted"` — 加入位置衰减、时间权重
3. 对比 `variant="top5_weighted"` — 不同的归一化方式

**核心公式**：
```
sim[i][j] += loc_alpha * decay^(|loc_diff|-1) / log(history_len + 1)

归一化：
  baseline/weighted: sim[i][j] / sqrt(cnt_i * cnt_j)
  top5_weighted:     sim[i][j] / (max^0.4 * min^0.6)
```

**面试重点**：
- ItemCF 的计算复杂度是什么？→ O(N * L^2)，N=用户数，L=平均历史长度
- 如何优化？→ 倒排索引、只计算热门物品对、稀疏矩阵

### 第 2 步：热门召回 (`src/recall/hot.py`)

最简单的召回策略，10 行代码。按全局点击量排序，加时间窗口过滤。

**面试追问**：热门召回的缺点是什么？→ 无法个性化，所有用户看到一样的结果。

### 第 3 步：Bi-network (`src/recall/binetwork.py`)

**核心公式**：
```
sim[i][j] += 1 / (log(users_of_i + 1) * log(items_of_user + 1))
```

IDF 加权的共现计数：物品 i 被越多用户点击（越热门），权重越低；用户 u 点击越多物品（越活跃），贡献越小。

**面试对比**：和 ItemCF 的区别？→ ItemCF 是用户维度的共现，Bi-network 是物品-用户-物品的二部图传播。

### 第 4 步：Swing (`src/recall/swing.py`)

**核心公式**：
```
sim[i][j] += 1 / (alpha + |co_click_users|)
alpha = 5.0
```

如果两个物品只有少数几个用户同时点击，说明这少数用户的行为更有意义（强信号）。如果大量用户同时点击，可能是热门效应（弱信号）。

**面试高频**：Swing 和 ItemCF 的核心区别？→ Swing 通过"用户对共交互图"引入了更严格的共现约束，能过滤掉热门噪声。

### 第 5 步：YouTubeDNN (`src/recall/youtube_dnn.py`)

**架构理解**：
```
User Tower:
  user_embedding(user_id) + mean_pool(hist_embedding(seq))
  → DNN(128, 64) → user_emb

Item Tower:
  item_embedding(item_id) → item_emb

Score = dot(user_emb, item_emb)
```

**关键流程**：
1. 训练阶段：滑动窗口生成正样本，随机采样负样本（1:4），BCE 损失
2. 推理阶段：提取所有 user/item embedding，Faiss ANN 检索 top-K

**面试必问**：
- 为什么用双塔而不是单塔？→ 双塔可以预计算 item embedding，在线只需计算 user embedding + ANN 检索
- 负采样怎么做？→ 随机从用户未点击的物品中采样
- Faiss 是什么？→ Facebook 的向量检索库，支持亿级向量的毫秒级检索

### 第 6 步：多路召回融合 (`src/recall/combiner.py`)

**流程**：
1. 每路召回的分数做 per-user MinMax 归一化到 [0, 1]
2. 乘以该路的权重
3. 同一物品多路出现时分数累加
4. 取 top-K

**面试追问**：为什么要 per-user 归一化？→ 不同召回通道的分数尺度不同（ItemCF 是共现计数，YouTubeDNN 是余弦相似度），归一化后才能公平融合。

---

## 阶段 4: 攻克排序模型 (2-3 天)

**核心文件**：`src/features.py`、`src/ranking/`、`src/negative_sampling.py`

### 第 1 步：特征工程 (`src/features.py`)

逐列理解每个特征：

| 特征类别 | 特征示例 | 含义 |
|---------|---------|------|
| 用户统计 | user_click_count, user_avg_words | 用户活跃度、阅读偏好 |
| 物品统计 | item_click_count, item_user_count | 物品热度 |
| 时间差 | candidate_created_diff | 候选文章 vs 最后点击文章的发布时间差 |
| 字数差 | word_diff_last, word_diff_avg | 候选文章 vs 用户偏好的字数差异 |
| 上下文 | hour, weekday, freshness | 点击时间上下文 |
| 交叉 | user_category_count | 用户对该类别的历史点击次数 |
| 相似度 | itemcf_sim_last | 候选与最后点击的 ItemCF 相似度 |

**面试高频**：哪些特征最重要？→ 通常时间差、相似度、物品热度是最强特征。

### 第 2 步：负采样 (`src/negative_sampling.py`)

| 策略 | 说明 | 使用方案 |
|------|------|---------|
| 召回即负样本 | 召回候选中非 ground-truth 的都是负样本 | S2 |
| 双轴采样 | 按用户 + 按物品各采样 1-5 个，取并集 | S3/S5 |

**面试追问**：正负样本比例不平衡怎么办？→ 负采样、Focal Loss、调整类别权重。

### 第 3 步：排序模型

**pointwise (`lgb_classifier.py`)**：
- 二分类：每个 (user, item) 独立预测点击概率
- 损失函数：Binary Cross-Entropy
- 优点：简单，实现容易
- 缺点：没有考虑同一用户候选之间的相对顺序

**listwise (`lgb_ranker.py`)**：
- 直接优化排序指标 NDCG
- 损失函数：LambdaRank
- 优点：直接优化最终评估指标
- 缺点：需要 group 信息（每个用户的候选为一组）

**面试重点**：pointwise / pairwise / listwise 三种排序思路的区别：
- pointwise：把排序当分类/回归，独立预测每个文档的分数
- pairwise：比较文档对的相对顺序（如 RankNet）
- listwise：直接优化整个列表的排序指标（如 LambdaRank、ListNet）

---

## 阶段 5: 横向对比 4 个方案 (1 天)

**核心文件**：`solutions/` 目录

| 维度 | S1 | S2 | S3 | S5 |
|------|----|----|----|-----|
| 召回路数 | 2 | 3 | 2 | 3 |
| 召回方法 | ItemCF + YouTubeDNN | ItemCF + BiNet + W2V | Hot + 30s-i2i | ItemCF + Swing + Item2Vec |
| 排序模型 | 无 | LGBClassifier | LGBRanker | LGBRanker |
| 负采样 | 随机 1:4 | 召回即负 | 双轴 | 双轴 |
| 交叉验证 | 无 | 5-fold GroupKFold | 80/20 用户划分 | 80/20 用户划分 |

**面试场景**："你这个项目用了哪些方法？哪个效果最好？为什么？"

**参考回答**：
> 我实现了 4 种方案。召回阶段效果最大的提升来自多路融合和高质量的相似度算法（如 Swing、YouTubeDNN）。排序阶段 LambdaRank 比 Classifier 略好，因为直接优化 NDCG。负采样策略对排序效果影响很大，双轴采样比随机采样更好。

---

## 阶段 6: 能讲清楚项目 (1 天)

### 30 秒版本

> 我复现了天池新闻推荐竞赛的 4 份获奖方案，包含 8 种召回算法和 2 种排序模型。数据用 Leave-One-Out 划分，评估指标是 Recall@K 和 NDCG@5。

### 3 分钟版本

> 整个系统分召回和排序两阶段。召回阶段我实现了 ItemCF、Bi-network、Swing、YouTubeDNN 等 8 种算法，通过多路融合取 top-50 候选。排序阶段用 LightGBM 对候选打分，对比了 pointwise (Classifier) 和 listwise (LambdaRank) 两种思路。特征工程包含用户统计、物品统计、时间差、相似度等 19 维特征。负采样用了双轴策略保证多样性。

### 深挖准备

| 问题 | 参考答案 |
|------|---------|
| ItemCF 的时间复杂度？如何加速？ | O(N * L^2)，用倒排索引优化到 O(N * L * K) |
| YouTubeDNN 为什么用双塔？ | 可以预计算 item embedding，在线只需 user embedding + ANN 检索 |
| LambdaRank 和 GBDT 的区别？ | LambdaRank 是 listwise 损失，直接优化 NDCG；GBDT 是 pointwise |
| 冷启动怎么处理？ | 热门兜底 + embedding 相似度 + 内容特征 |
| 如何处理数据稀疏？ | 负采样、embedding 降维、引入 side information |
| 多路召回融合的权重怎么定？ | 离线评估各路召回率，按效果手动调参或自动调参 |

---

## 学习检查清单

完成以下检查说明你已掌握这个项目：

- [ ] 能独立画出 S1 方案的完整流程图（数据加载 → LOO 划分 → ItemCF 召回 → YouTubeDNN 召回 → 融合 → 评估）
- [ ] 能解释 ItemCF 的加权公式中每个因子的含义
- [ ] 能说清楚 YouTubeDNN 双塔的训练和推理流程
- [ ] 能对比 pointwise 和 listwise 排序的区别
- [ ] 能解释为什么需要负采样以及两种负采样策略的区别
- [ ] 能说出 4 个方案的核心差异
- [ ] 能用 30 秒和 3 分钟两个版本介绍这个项目
