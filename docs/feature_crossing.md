# 特征交叉模型技术讨论

本文档讨论在 LightGBM Ranker 基线之上，使用深度特征交叉模型（DCN、DeepFM、AutoTINT、RankMixer）进行排序的可行性与技术方案。

---

## 1. LightGBM Ranker 基线

### 当前实现

- **模型**: `lgb.LGBMRanker` + LambdaRank 目标
- **损失**: 直接优化 NDCG（listwise）
- **特征**: 19 维手工特征（用户统计、物品统计、时间差、文本差异、上下文、交叉、召回分数）
- **优势**: 训练快、可解释性强、对小数据友好

### 局限

- 特征交叉依赖手工设计（仅 `user_category_count` 一维显式交叉）
- 无法自动发现高阶特征组合
- 对 embedding 类特征利用不充分（仅 `emb_sim_last` 一维余弦相似度）

---

## 2. DCN (Deep & Cross Network)

### 核心思想

Cross Network 通过多层交叉操作，显式学习有限阶（d 阶）的特征组合，每一层在前一层输出与原始输入之间做 element-wise 乘法，复杂度仅增加 `O(d * d)`。

### 网络结构

```
输入特征 x₀
    ├── Cross Network (d layers)
    │   x_{l+1} = x₀ * (wᵀ * x_l + b) + x_l
    │   每层增加一阶交叉，d 层捕获 d+1 阶组合
    └── Deep Network (多层 MLP)
        自动学习任意阶特征组合

Concat(cross_out, deep_out) -> 全连接 -> sigmoid
```

### DCN-V1 vs V2

| 维度 | DCN-V1 | DCN-V2 (Mix of Low-rank) |
|------|--------|--------------------------|
| 交叉矩阵 | 全秩 `d × d` | 低秩分解 `W = U * Vᵀ`，参数量从 `d²` 降到 `d * r + d * r` |
| 混合架构 | 无 | 可选 MoE，多个 cross 层并行后 gating |
| 参数效率 | 高阶时参数爆炸 | 低秩 + MoE 大幅降低参数量 |

### 对接当前项目

```python
# 特征输入设计
dense_features = [所有 19 维数值特征]
sparse_features = [category_id]  # 需 embedding

# 输入 = [dense_features || embedding(category_id) || embedding(user_id) || embedding(article_id)]
# Cross Network 直接在拼接后的向量上做交叉
```

### 面试要点

- **为什么用 Cross 而不是纯 DNN?** Cross 显式保证学习到有限阶交叉，DNN 的交叉是隐式的、无法保证阶数
- **V2 低秩的意义?** 实际排序场景特征维度不高（~100），全秩矩阵参数量可控；但在广告场景特征维度达数千时，低秩是必须的

---

## 3. DeepFM

### 核心思想

将 FM（Factorization Machine）的二阶交叉能力与 DNN 的高阶交叉能力并行组合。FM 层无需手工特征工程即可自动学习所有二阶特征交叉。

### 网络结构

```
输入: 稀疏特征 embedding + 稠密特征
    ├── FM Layer
    │   一阶: wᵢ * xᵢ (线性部分)
    │   二阶: Σᵢ Σⱼᵢ <vᵢ, vⱼ> * xᵢ * xⱼ
    │   高效计算: 0.5 * [(<Σvᵢxᵢ>² - Σ<vᵢxᵢ>²)]
    │   时间复杂度: O(k * n)，k=embedding 维度，n=特征数
    └── Deep Layer (MLP)
        自动学习高阶特征组合

FM_out + Deep_out -> sigmoid
```

### 与 DCN 对比

| 维度 | DeepFM | DCN |
|------|--------|-----|
| 二阶交叉 | FM 层显式、高效 | Cross 第 1 层隐式学习 |
| 高阶交叉 | DNN 隐式 | Cross 显式（有限阶） |
| 输入要求 | 稠密+稀疏均可 | 主要面向稠密特征 |
| 参数效率 | FM 二阶参数 O(n*k) | Cross 参数 O(d²) |

### 对接当前项目

```python
# 稀疏特征
sparse_features = [user_id, article_id, category_id, hour, weekday]
# 每个 sparse feature 独立 embedding

# 稠密特征
dense_features = [score, user_click_count, freshness, ...]  # 剩余 14 维

# FM Layer 自动对所有 sparse feature 做二阶交叉
# Deep Layer 对拼接后的全量特征做高阶交叉
```

### 面试要点

- **FM 二阶交叉为什么高效?** 利用 `(<Σvᵢxᵢ>² - Σ<vᵢxᵢ>²)` 恒等式，将 O(n²k) 降到 O(nk)
- **DeepFM vs Wide&Deep?** Wide&Deep 的 Wide 部分需要手工设计交叉特征，DeepFM 的 FM 部分自动学习

---

## 4. AutoTINT (Automatic Task-Interaction Network)

### 核心思想

通过 Attention 机制自动发现特征之间的交互模式，不需要预定义交叉结构。每个特征作为 token，self-attention 让模型自动学习哪些特征对之间应该交互。

### 网络结构

```
输入特征 -> Embedding Layer -> [e₁, e₂, ..., eₙ] (特征 token 序列)
    ├── Self-Attention Layers (L layers)
    │   Q = Wq * E, K = Wk * E, V = Wv * E
    │   Attention = softmax(QKᵀ / √d) * V
    │   每个特征 token 关注所有其他特征，自动发现重要交互
    └── 也可加入 task-specific attention (任务感知)
        用任务 embedding 做 query，特征做 key/value

Attention_output -> MLP -> sigmoid
```

### 与 DNN-based 交叉的区别

| 维度 | AutoTINT / Attention | DCN / DeepFM |
|------|---------------------|--------------|
| 交叉方式 | 全局 attention，所有特征对都能交互 | 层级式，逐层增加交叉阶数 |
| 可解释性 | attention 权重可可视化交互强度 | Cross 权重不易解读 |
| 计算复杂度 | O(n² * d) | DCN O(n*d)，FM O(n*k) |
| 特征数量敏感 | 特征少时（<50）表现好 | 特征多时更高效 |

### 对接当前项目

```python
# 将每个特征 embedding 化
# user_click_count -> embedding(离散化后的 bin)
# category_id -> embedding
# score -> linear projection -> token

# 19 个特征 -> 19 个 token -> 2-layer self-attention
# 注意力权重可分析：哪些特征交互最重要
```

### 面试要点

- **为什么用 Attention 做特征交叉?** 传统方法需要预定义交叉结构（FM 二阶、Cross 有限阶），Attention 自动发现任意阶交互
- **计算开销问题?** 对于少量特征（<50），O(n²) 可接受；大量特征时需要 sparse attention 或 factorization

---

## 5. RankMixer

### 核心思想

将 MLP-Mixer（CV 领域替代 Transformer 的高效架构）引入推荐排序。用 token-mixing 和 channel-ming 替代 self-attention，在保持表达力的同时降低计算复杂度。

### 网络结构

```
输入特征 -> Embedding Layer -> [e₁, e₂, ..., eₙ] (特征 token 序列, shape: n × d)
    ├── Token-Mixing (特征间交互)
    │   对每个 channel 维度，跨所有 token 做 MLP
    │   output = MLP_col(input_transposed)  # shape: n × d -> n × d
    │   等价于：每个特征对所有其他特征的线性组合
    └── Channel-Mixing (特征内变换)
        对每个 token，跨所有 channel 做 MLP
        output = MLP_row(input)  # shape: n × d -> n × d

L 个 Mixer Block 堆叠 -> Flatten -> MLP -> sigmoid
```

### 与 Attention 对比

| 维度 | RankMixer | Self-Attention (AutoTINT) |
|------|-----------|--------------------------|
| 特征交互方式 | Token-mixing MLP（线性组合） | QKᵀ 点积（非线性） |
| 复杂度 | O(n * d²) | O(n² * d) |
| 参数量 | 中等 | 较多（Q/K/V 矩阵） |
| 长序列 | 优势明显（n 大时） | n² 瓶颈 |
| 表达力 | 局部交互为主 | 全局交互 |

### 对接当前项目

```python
# 特征 token 化（同 AutoTINT）
# 19 个特征 -> 19 个 token

# Mixer Block x 2
# Token-Mixing: 捕获 "user_click_count 和 category_id 的关系"
# Channel-Mixing: 捕获 "embedding 维度间的非线性变换"

# 输出: MLP -> sigmoid -> 预测分数
```

### 面试要点

- **Mixer vs Attention 选哪个?** 特征数少（<50）时两者差异不大，Attention 表达力更强；特征数多时 Mixer 更高效
- **为什么不用 Transformer?** Transformer 的 self-attention 对排序场景来说是 overkill，Mixer 用更简单的结构达到类似效果

---

## 6. 对比总结

| 模型 | 交叉阶数 | 参数量 | 训练速度 | 可解释性 | 适用场景 |
|------|---------|--------|---------|---------|---------|
| **LightGBM** | 2-3 阶（树深度） | 小 | 快 | 高（特征重要度） | 基线、小数据、快速迭代 |
| **DCN-V2** | 有限阶（d+1） | 中 | 中 | 中（cross 权重） | 大规模广告/推荐 |
| **DeepFM** | 2 阶显式 + 高阶隐式 | 中 | 中 | 中（FM 权重） | 稀疏特征多的场景 |
| **AutoTINT** | 任意阶 | 较大 | 慢 | 高（attention 权重） | 特征数少、需要可解释性 |
| **RankMixer** | 任意阶 | 中 | 较快 | 低 | 特征数多、效率敏感 |

### 在当前项目的实验路线建议

```
Phase 1: LightGBM Ranker 基线（已实现）
   ↓
Phase 2: DeepFM（实现最简单，FM 层与现有特征天然兼容）
   ↓
Phase 3: DCN-V2（Cross Network 对稠密特征效果好）
   ↓
Phase 4: RankMixer / AutoTINT（前沿探索，适合面试展示技术深度）
```

### 关键实验指标

- **离线**: AUC, NDCG@5（与 LightGBM 基线对比提升幅度）
- **效率**: 训练时间、推理延迟、参数量
- **可解释性**: 特征重要度 / attention 权重分析
