"""Multi-channel recall fusion with per-user MinMax normalization."""
from collections import defaultdict


def normalize_user_scores(item_score_list):
    """Per-user MinMax normalization: (score - min) / (max - min) + eps."""
    if not item_score_list:
        return item_score_list
    scores = [s for _, s in item_score_list]
    s_min, s_max = min(scores), max(scores)
    denom = s_max - s_min + 1e-8
    return [(iid, (s - s_min) / denom + 1e-3) for iid, s in item_score_list]


def combine_recall_results(multi_recall_dict, weight_dict=None, topk=50):
    """Merge multiple recall channels with weighted per-user MinMax scores.

    Args:
        multi_recall_dict: {channel_name: {user_id: [(item_id, score), ...]}}
        weight_dict: {channel_name: weight} or None (equal weights)
        topk: final candidates per user

    Returns:
        {user_id: [(item_id, score), ...]}
    """
    if weight_dict is None:
        weight_dict = {ch: 1.0 for ch in multi_recall_dict}

    merged = defaultdict(lambda: defaultdict(float))

    for channel, recall_dict in multi_recall_dict.items():
        w = weight_dict.get(channel, 1.0)
        for uid, items in recall_dict.items():
            normalized = normalize_user_scores(items)
            for iid, score in normalized:
                merged[uid][iid] += w * score

    result = {}
    for uid, item_scores in merged.items():
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        result[uid] = sorted_items[:topk]
    return result
