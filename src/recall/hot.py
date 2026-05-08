"""Hot/trending article recall with time window filtering (from top3).

For each user, returns globally popular articles that were created
within a time window before the user's last click.
"""
from collections import Counter
from src.recall.base import BaseRecall


class HotRecall(BaseRecall):
    name = "hot"

    def __init__(self, lag_hour_min=3, lag_hour_max=27):
        self.lag_hour_min = lag_hour_min
        self.lag_hour_max = lag_hour_max

    def build_similarity(self, user_item_time_dict, **kwargs):
        """No similarity matrix needed; returns click counts."""
        item_counts = Counter()
        for uid, items in user_item_time_dict.items():
            for iid, _ in items:
                item_counts[iid] += 1
        return item_counts

    def recommend(self, user_id, user_items, sim, topk=10,
                  item_created_abs_time_dict=None, **kwargs):
        """Return popular articles within time window, excluding clicked items.

        Time window: [last_click - lag_hour_max, last_click - lag_hour_min]
        """
        if not user_items:
            return []

        clicked = {iid for iid, _ in user_items}
        last_ts = user_items[-1][1]
        result = []

        for iid, count in sim.most_common():
            if iid in clicked:
                continue
            # Time window filter
            if item_created_abs_time_dict:
                created_ts = item_created_abs_time_dict.get(iid, 0)
                hours_diff = (last_ts - created_ts) / (1000 * 3600)
                if hours_diff < self.lag_hour_min or hours_diff > self.lag_hour_max:
                    continue
            result.append((iid, float(count)))
            if len(result) >= topk:
                break
        return result
