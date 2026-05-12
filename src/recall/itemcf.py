"""ItemCF recall with 3 variants: baseline, weighted, top5_weighted.

Variants differ in:
- Similarity weight factors (position, click time, creation time)
- Normalization method (sqrt vs popularity-penalized)
"""
import math
import os
import pickle
import numpy as np
from collections import defaultdict
from src.recall.base import BaseRecall


class ItemCFRecall(BaseRecall):
    name = "itemcf"

    def __init__(self, variant="weighted"):
        """
        Args:
            variant: "baseline" | "weighted" | "top5_weighted"
        """
        self.variant = variant

    def build_similarity(self, user_item_time_dict, item_created_time_dict=None,
                         save_path=None, use_cache=True, **kwargs):
        """Build item-item similarity matrix.

        Returns:
            i2i_sim: {item_i: {item_j: similarity_score}}
        """
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached ItemCF sim: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        print(f"Building ItemCF similarity (variant={self.variant})...")
        i2i_count = defaultdict(lambda: defaultdict(float))
        item_cnt = defaultdict(int)

        for uid, item_time_list in user_item_time_dict.items():
            n = len(item_time_list)
            for i in range(n):
                item_i, time_i = item_time_list[i]
                item_cnt[item_i] += 1

                for j in range(n):
                    if i == j:
                        continue
                    item_j, time_j = item_time_list[j]

                    if self.variant == "baseline":
                        i2i_count[item_i][item_j] += 1.0 / math.log(1 + n)  # 惩罚高活跃用户
                    else:
                        # Position direction weight
                        loc_alpha = 1.0 if j > i else 0.7
                        # Position decay
                        decay = 0.8 if self.variant == "top5_weighted" else 0.9
                        loc_weight = loc_alpha * (decay ** (abs(j - i) - 1))

                        w = loc_weight / math.log(1 + n)

                        if item_created_time_dict and self.variant != "baseline":
                            ct_i = item_created_time_dict.get(item_i, 0.5)
                            ct_j = item_created_time_dict.get(item_j, 0.5)
                            created_time_weight = np.exp(0.8 ** abs(ct_i - ct_j))
                            w *= created_time_weight

                        i2i_count[item_i][item_j] += w

        # Normalize
        # 同样的共现次数 5 次，冷门物品之间的相似度远高于热门物品之间的。
        # 这正是我们想要的——冷门物品的共现更能说明真实的兴趣关联，热门物品的共现更多是"碰巧都点了"。
        i2i_sim = defaultdict(dict)
        for item_i, related in i2i_count.items():
            for item_j, wij in related.items():
                if self.variant == "top5_weighted":
                    # Asymmetric popularity penalty
                    cnt_i = item_cnt[item_i]
                    cnt_j = item_cnt[item_j]
                    tmp_max = max(cnt_i, cnt_j)
                    tmp_min = min(cnt_i, cnt_j)
                    i2i_sim[item_i][item_j] = wij / (
                        (tmp_max ** 0.4) * (tmp_min ** 0.6) + 1e-8
                    )
                else:
                    i2i_sim[item_i][item_j] = wij / math.sqrt(
                        item_cnt[item_i] * item_cnt[item_j] + 1e-8
                    )

        i2i_sim = dict(i2i_sim)
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(i2i_sim, f)
            print(f"Cached ItemCF sim: {save_path}")

        return i2i_sim

    def recommend(self, user_id, user_items, sim, topk=50,
                  item_topk_click=None, **kwargs):
        """Generate recall candidates using ItemCF.

        Args:
            user_items: [(item_id, timestamp), ...] sorted by time
            sim: i2i_sim dict from build_similarity
            topk: number of candidates
            item_topk_click: popular items for fallback
        """
        rank = defaultdict(float)
        # Use last 2 items with recency decay
        recent = user_items[-2:][::-1]
        for loc, (iid, _) in enumerate(recent):
            if iid not in sim:
                continue
            for related_iid, wij in sorted(
                sim[iid].items(), key=lambda x: x[1], reverse=True
            )[:200]:
                rank[related_iid] += wij * (0.7 ** loc)

        # Remove already clicked
        clicked = {iid for iid, _ in user_items}
        rank = {iid: s for iid, s in rank.items() if iid not in clicked}

        result = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:topk]

        # Fallback: fill with popular items
        if len(result) < topk and item_topk_click:
            seen = {iid for iid, _ in result} | clicked
            for iid in item_topk_click:
                if iid not in seen:
                    result.append((iid, -1.0))
                    if len(result) >= topk:
                        break
        return result
