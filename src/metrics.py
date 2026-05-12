"""Evaluation metrics for recall and ranking stages."""
import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_k(recall_dict, test_labels, k):
    """Fraction of users whose true item appears in top-k recall.

    Args:
        recall_dict: {user_id: [(item_id, score), ...]} or {user_id: [item_id, ...]}
        test_labels: {user_id: true_item_id}
        k: cutoff
    """
    hits, total = 0, 0
    for uid, true_item in test_labels.items():
        if uid not in recall_dict:  # 冷启动用户 如何处理？
            total += 1
            continue
        candidates = recall_dict[uid]
        if isinstance(candidates[0], tuple):
            candidates = [x[0] for x in candidates]
        if true_item in candidates[:k]:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def mrr_at_k(recall_dict, test_labels, k=5):  # 平均倒数排名（Mean Reciprocal Rank,MRR）
    """Mean Reciprocal Rank at k.

    Args:
        recall_dict: {user_id: [(item_id, score), ...]} or {user_id: [item_id, ...]}
        test_labels: {user_id: true_item_id}
        k: cutoff
    """
    rr_sum, total = 0.0, 0
    for uid, true_item in test_labels.items():
        if uid not in recall_dict:
            total += 1
            continue
        candidates = recall_dict[uid]
        if isinstance(candidates[0], tuple):
            candidates = [x[0] for x in candidates]
        for rank, item in enumerate(candidates[:k], 1):
            if item == true_item:
                rr_sum += 1.0 / rank
                break   # 只看一个正确答案
        total += 1
    return rr_sum / total if total > 0 else 0.0

def evaluate_recall(recall_dict, test_labels, k_list=(5, 10, 20, 50)):
    """Evaluate recall and MRR at multiple k values."""
    results = {}
    for k in k_list:
        results[f"Recall@{k}"] = recall_at_k(recall_dict, test_labels, k)
    results["MRR@5"] = mrr_at_k(recall_dict, test_labels, 5)
    print("Recall metrics:")
    for name, val in results.items():
        print(f"  {name}: {val:.4f}")
    return results

# AUC 看全局区分，NDCG 看头部排序。
def evaluate_ranking(y_true, y_pred, user_ids=None, k=5):
    """Evaluate ranking with AUC and NDCG@k.

    Args:
        y_true: array of labels (0/1)
        y_pred: array of predicted scores
        user_ids: array of user_ids for NDCG grouping (optional)
        k: NDCG cutoff
    """
    results = {}
    try:
        results["AUC"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        results["AUC"] = 0.0

    if user_ids is not None:
        results[f"NDCG@{k}"] = _ndcg_at_k(y_true, y_pred, user_ids, k)
    else:
        results[f"NDCG@{k}"] = 0.0

    print("Ranking metrics:")
    for name, val in results.items():
        print(f"  {name}: {val:.4f}")
    return results


def _ndcg_at_k(y_true, y_pred, user_ids, k=5):  # Normalized Discounted Cumulative Gain 归一化折损累计增益
    """Compute mean NDCG@k grouped by user_id."""
    from collections import defaultdict
    user_data = defaultdict(list)
    for uid, label, score in zip(user_ids, y_true, y_pred):
        user_data[uid].append((label, score))

    ndcg_sum, count = 0.0, 0
    for uid, items in user_data.items():
        items.sort(key=lambda x: x[1], reverse=True)
        dcg = sum(
            items[i][0] / np.log2(i + 2) for i in range(min(k, len(items)))
        )
        ideal = sorted(items, key=lambda x: x[0], reverse=True)
        idcg = sum(
            ideal[i][0] / np.log2(i + 2) for i in range(min(k, len(ideal)))
        )
        if idcg > 0:
            ndcg_sum += dcg / idcg
            count += 1
    return ndcg_sum / count if count > 0 else 0.0
