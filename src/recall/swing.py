"""Swing algorithm (from top5).

For each pair of users (u1, u2) who co-clicked items:
    sim[i][j] += 1 / (alpha + |co_clicked_items|)

Filters out users with >max_hist or <min_hist clicks for memory efficiency.
"""
import math
import os
import pickle
from collections import defaultdict
from src.recall.base import BaseRecall


class SwingRecall(BaseRecall):
    name = "swing"

    def __init__(self, alpha=5.0, max_user_hist=8, min_user_hist=3):
        self.alpha = alpha
        self.max_user_hist = max_user_hist
        self.min_user_hist = min_user_hist

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, **kwargs):
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached Swing sim: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        print("Building Swing similarity...")
        # Build item -> users mapping, filter extreme users
        item_users = defaultdict(set)
        filtered_users = set()
        for uid, items in user_item_time_dict.items():
            if self.min_user_hist <= len(items) <= self.max_user_hist:
                filtered_users.add(uid)
                for iid, _ in items:
                    item_users[iid].add(uid)

        # Count items
        item_cnt = defaultdict(int)
        for iid, users in item_users.items():
            item_cnt[iid] = len(users)

        # Build user pair -> shared items
        u_u_items = defaultdict(list)
        for iid, users in item_users.items():
            users_list = list(users)
            for a in range(len(users_list)):
                for b in range(a + 1, len(users_list)):
                    u1, u2 = users_list[a], users_list[b]
                    u_u_items[(u1, u2)].append(iid)

        # Compute Swing scores
        sim = defaultdict(lambda: defaultdict(float))
        for (u1, u2), co_items in u_u_items.items():
            n_co = len(co_items)
            for i in co_items:
                for j in co_items:
                    if i != j:
                        sim[i][j] += 1.0 / (self.alpha + n_co)

        # Normalize
        result = defaultdict(dict)
        for i, related in sim.items():
            for j, score in related.items():
                result[i][j] = score / math.sqrt(item_cnt[i] * item_cnt[j] + 1e-8)
        result = dict(result)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Cached Swing sim: {save_path}")
        return result

    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        rank = defaultdict(float)
        clicked = {iid for iid, _ in user_items}
        recent = user_items[-2:][::-1]

        for loc, (iid, _) in enumerate(recent):
            if iid not in sim:
                continue
            for related_iid, score in sorted(
                sim[iid].items(), key=lambda x: x[1], reverse=True
            )[:200]:
                if related_iid not in clicked:
                    rank[related_iid] += score * (0.7 ** loc)

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:topk]
