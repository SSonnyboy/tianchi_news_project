"""30-second interval i2i similarity (from top3).

Detects consecutive clicks exactly 30000ms apart and counts article pair co-occurrences.
This exploits a domain-specific pattern in the competition dataset.
"""
import os
import pickle
from collections import defaultdict
from src.recall.base import BaseRecall


class I2I30sRecall(BaseRecall):
    name = "i2i_30s"

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, **kwargs):
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached 30s-i2i sim: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        print("Building 30s-interval i2i similarity...")
        sim = defaultdict(lambda: defaultdict(int))

        for uid, item_time_list in user_item_time_dict.items():
            for i in range(1, len(item_time_list)):
                prev_iid, prev_ts = item_time_list[i - 1]
                curr_iid, curr_ts = item_time_list[i]
                if curr_ts - prev_ts == 30000:
                    sim[prev_iid][curr_iid] += 1
                    sim[curr_iid][prev_iid] += 1

        # Convert to sorted format
        result = {}
        for iid, related in sim.items():
            sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
            result[iid] = {
                "sorted_keys": [k for k, _ in sorted_related],
                "related_arts": dict(related),
            }

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Cached 30s-i2i sim: {save_path}")
        return result

    def recommend(self, user_id, user_items, sim, topk=25,
                  item_created_abs_time_dict=None, min_count=2, **kwargs):
        """Recall using 30s-interval similarity with time window filter.

        Time window: articles created within 0-27 hours before last click.
        """
        if not user_items:
            return []

        clicked = {iid for iid, _ in user_items}
        last_ts = user_items[-1][1]
        rank = defaultdict(float)

        for iid, _ in user_items:
            if iid not in sim:
                continue
            related = sim[iid]
            for related_iid in related["sorted_keys"][:100]:
                if related_iid in clicked:
                    continue
                count = related["related_arts"].get(related_iid, 0)
                if count < min_count:
                    continue
                # Time window filter: 0-27 hours before last click
                if item_created_abs_time_dict:
                    created_ts = item_created_abs_time_dict.get(related_iid, 0)
                    hours_diff = (last_ts - created_ts) / (1000 * 3600)
                    if hours_diff < 0 or hours_diff > 27:
                        continue
                rank[related_iid] += count

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:topk]
