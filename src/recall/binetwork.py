"""Bi-network item-item similarity (from top2).

For each item i, for each user u who clicked i, for each item j that u clicked:
    sim[i][j] += 1 / (log(|users_of_i|+1) * log(|items_of_u|+1))

This is an IDF-weighted co-occurrence count.
"""
import math
import os
import pickle
from collections import defaultdict
from src.recall.base import BaseRecall


class BiNetworkRecall(BaseRecall):
    name = "binetwork"

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, **kwargs):
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached BiNetwork sim: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        print("Building BiNetwork similarity...")
        # Build item -> users mapping
        item_users = defaultdict(set)
        user_items = defaultdict(set)
        for uid, item_time_list in user_item_time_dict.items():
            for iid, _ in item_time_list:
                item_users[iid].add(uid)
                user_items[uid].add(iid)

        sim = defaultdict(dict)
        for item_i, users in item_users.items():
            for uid in users:
                for item_j in user_items[uid]:
                    if item_i == item_j:
                        continue
                    sim[item_i][item_j] = sim[item_i].get(item_j, 0) + 1.0 / (
                        math.log(len(users) + 1) * math.log(len(user_items[uid]) + 1)
                    )

        sim = dict(sim)
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(sim, f)
            print(f"Cached BiNetwork sim: {save_path}")
        return sim

    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        """Use last 1 clicked item for recall."""
        rank = defaultdict(float)
        clicked = {iid for iid, _ in user_items}

        # Use last item
        if user_items:
            last_iid = user_items[-1][0]
            if last_iid in sim:
                for related_iid, score in sorted(
                    sim[last_iid].items(), key=lambda x: x[1], reverse=True
                )[:100]:
                    if related_iid not in clicked:
                        rank[related_iid] = score

        result = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:topk]
        return result
