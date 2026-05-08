"""Negative sampling strategies for ranking stage training."""
import numpy as np
import pandas as pd


def recall_as_negatives(recall_df, test_labels):
    """Top2 strategy: all recall candidates become training data.

    Positive: the user's true test item (if in recall set)
    Negative: all other recall candidates

    Args:
        recall_df: DataFrame with [user_id, article_id, score]
        test_labels: {user_id: true_article_id}

    Returns:
        DataFrame with [user_id, article_id, label]
    """
    recall_df = recall_df.copy()
    recall_df["label"] = 0
    for uid, true_item in test_labels.items():
        mask = (recall_df["user_id"] == uid) & (recall_df["article_id"] == true_item)
        recall_df.loc[mask, "label"] = 1
    return recall_df


def dual_axis_negative_sampling(recall_df, test_labels, min_samples=1,
                                 max_samples=5, seed=42):
    """Top3/Top5 strategy: sample negatives by both user and item axis.

    1. Label positive samples from test_labels
    2. Per-user: sample min-max negatives per user
    3. Per-item: sample min-max negatives per item
    4. Union + deduplicate + combine with all positives

    Args:
        recall_df: DataFrame with [user_id, article_id, score]
        test_labels: {user_id: true_article_id}
        min_samples: minimum negatives per group
        max_samples: maximum negatives per group

    Returns:
        DataFrame with [user_id, article_id, label]
    """
    rng = np.random.RandomState(seed)
    recall_df = recall_df.copy()

    # Label positives
    recall_df["label"] = 0
    for uid, true_item in test_labels.items():
        mask = (recall_df["user_id"] == uid) & (recall_df["article_id"] == true_item)
        recall_df.loc[mask, "label"] = 1

    pos_df = recall_df[recall_df["label"] == 1]
    neg_df = recall_df[recall_df["label"] == 0]

    if len(neg_df) == 0:
        return pos_df

    def sample_group(group):
        n = len(group)
        k = min(max(int(n * 0.001), min_samples), max_samples, n)
        return group.sample(n=k, random_state=rng)

    # Per-user sampling
    neg_user = neg_df.groupby("user_id", group_keys=False).apply(sample_group)
    # Per-item sampling
    neg_item = neg_df.groupby("article_id", group_keys=False).apply(sample_group)

    # Union + deduplicate
    neg_sampled = pd.concat([neg_user, neg_item]).drop_duplicates(
        subset=["user_id", "article_id"]
    )

    result = pd.concat([pos_df, neg_sampled], ignore_index=True)
    print(f"Negative sampling: {len(pos_df)} pos + {len(neg_sampled)} neg = {len(result)} total")
    return result
