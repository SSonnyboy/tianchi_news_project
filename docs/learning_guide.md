# 学习路线：面向简历面试的推荐系统代码库

## 概述

本指南帮助你从零开始啃下这份推荐系统代码库，按"由浅入深、先跑通再理解"的原则组织。每一步都对应面试高频考点。

建议总时长：7 天（每天 2-3 小时）。

---

## 阶段 1: 跑通全链路 (1 天)

**目标**：先跑起来，建立直觉。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 把数据文件放到 data/ 目录
#    - train_click_log.csv
#    - articles.csv
#    - articles_emb.csv

# 3. 运行统一方案
python -c "
from solutions.solution_unified import SolutionUnified
pipeline = SolutionUnified()
metrics = pipeline.run()
"
```

观察输出的 Recall@5、NDCG@5 是多少，不需要理解代码细节。

**面试价值**：能说清"我实现了一个完整的推荐系统召回+排序 pipeline"。

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

## 阶段 3: 攻克召回算法 (2-3 天)

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

### 第 2 步：热门召回 + 30s-i2i (`src/recall/hot.py`, `src/recall/i2i_30s.py`)

热门召回：最简单的召回策略，10 行代码。按全局点击量排序，加时间窗口过滤。
30s-i2i：统计恰好间隔 30 秒点击的文章对共现次数，利用竞赛数据的领域特定模式。

**面试追问**：热门召回的缺点是什么？→ 无法个性化，所有用户看到一样的结果。

### 第 3 步：Swing (`src/recall/swing.py`)

**核心公式**：
```
sim[i][j] += 1 / (alpha + |co_click_users|)
alpha = 5.0
```

如果两个物品只有少数几个用户同时点击，说明这少数用户的行为更有意义（强信号）。如果大量用户同时点击，可能是热门效应（弱信号）。

**面试高频**：Swing 和 ItemCF 的核心区别？→ Swing 通过"用户对共交互图"引入了更严格的共现约束，能过滤掉热门噪声。

### 第 4 步：YouTubeDNN (`src/recall/youtube_dnn.py`)

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

### 第 5 步：多路召回融合 (`src/recall/combiner.py`)

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

使用双轴采样策略：按用户 + 按物品各采样 1-5 个负样本，取并集。保证正负样本多样性。

**面试追问**：正负样本比例不平衡怎么办？→ 负采样、Focal Loss、调整类别权重。

### 第 3 步：排序模型 (`src/ranking/lgb_ranker.py`)

**LightGBM LambdaRanker**：
- 直接优化排序指标 NDCG
- 损失函数：LambdaRank（listwise）
- 优点：直接优化最终评估指标
- 需要 group 信息（每个用户的候选为一组）

**面试重点**：pointwise / pairwise / listwise 三种排序思路的区别：
- pointwise：把排序当分类/回归，独立预测每个文档的分数
- pairwise：比较文档对的相对顺序（如 RankNet）
- listwise：直接优化整个列表的排序指标（如 LambdaRank、ListNet）

### 第 4 步：特征交叉模型探索

详见 `docs/feature_crossing.md`，讨论 DCN、DeepFM、AutoTINT、RankMixer 四种深度特征交叉模型：

| 模型 | 核心思路 | 面试关键词 |
|------|---------|-----------|
| DCN-V2 | Cross Network 显式有限阶交叉 | 低秩分解、MoE |
| DeepFM | FM 二阶 + DNN 高阶并行 | 自动特征工程、无需手工交叉 |
| AutoTINT | Self-Attention 自动发现交互 | 可解释性、全局交互 |
| RankMixer | MLP-Mixer token-mixing | 高效替代 Attention |

---

## 阶段 5: 能讲清楚项目 (1 天)

### 30 秒版本

> 我实现了一个新闻推荐系统，召回阶段用 ItemCF、Swing、YouTubeDNN、冷启动 4 路融合，排序阶段用 LightGBM LambdaRanker，并探索了 DCN、DeepFM 等深度特征交叉模型。

### 3 分钟版本

> 整个系统分召回和排序两阶段。召回阶段我实现了 4 路融合：ItemCF（基于点击共现加权）、Swing（基于用户对共交互模式）、YouTubeDNN（双塔神经网络 + Faiss 检索）、冷启动通道（热门 + 30s-i2i）。排序阶段用 LightGBM LambdaRanker 直接优化 NDCG，特征工程包含用户统计、物品统计、时间差、相似度等 19 维特征，负采样用双轴策略。在此基础上，我研究了 DCN、DeepFM、AutoTINT、RankMixer 等深度特征交叉模型，分析了它们在显式交叉阶数、参数效率、可解释性上的 trade-off。

### 深挖准备

| 问题 | 参考答案 |
|------|---------|
| ItemCF 的时间复杂度？如何加速？ | O(N * L^2)，用倒排索引优化到 O(N * L * K) |
| YouTubeDNN 为什么用双塔？ | 可以预计算 item embedding，在线只需 user embedding + ANN 检索 |
| LambdaRank 和 GBDT 的区别？ | LambdaRank 是 listwise 损失，直接优化 NDCG；GBDT 是 pointwise |
| 冷启动怎么处理？ | 热门兜底 + 30s-i2i 领域模式 + embedding 相似度 |
| 多路召回融合的权重怎么定？ | 离线评估各路召回率，按效果手动调参 |
| DCN 和 DeepFM 的核心区别？ | DCN 用 Cross Network 做显式有限阶交叉，DeepFM 用 FM 做二阶 + DNN 做高阶 |

---

## 学习检查清单

完成以下检查说明你已掌握这个项目：

- [ ] 能独立画出完整流程图（数据加载 → LOO 划分 → 4 路召回 → 融合 → 特征工程 → 排序 → 评估）
- [ ] 能解释 ItemCF 的加权公式中每个因子的含义
- [ ] 能说清楚 YouTubeDNN 双塔的训练和推理流程
- [ ] 能对比 Swing 和 ItemCF 的区别
- [ ] 能解释 LambdaRank 的 listwise 思路
- [ ] 能说明 DCN、DeepFM、RankMixer 的核心区别
- [ ] 能用 30 秒和 3 分钟两个版本介绍这个项目
