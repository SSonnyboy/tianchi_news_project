"""Data loading, Leave-One-Out split, and dictionary builders."""
import numpy as np
import pandas as pd
from collections import defaultdict
from src.config import (
    TRAIN_CLICK_PATH, ARTICLES_PATH, ARTICLES_EMB_PATH,
    COL_USER, COL_ITEM, COL_TIME, COL_CREATED, COL_WORDS, COL_CATEGORY,
    CONTEXT_COLS, RANDOM_SEED,
)
from src.utils import reduce_mem, Timer


def load_raw_data():
    """Load raw CSVs with optimized dtypes.

    Returns:
        click_df: click log (user_id, click_article_id, click_timestamp, context features)
        article_df: article metadata (article_id, category_id, created_at_ts, words_count)
        article_emb_df: article embeddings (article_id, emb_0..emb_249) or None
    """
    # 这种操作读取占用内存小 速度快
    click_dtypes = {
        COL_USER: "int32", COL_ITEM: "int32", COL_TIME: "int64",
        "click_environment": "int8", "click_deviceGroup": "int8",
        "click_os": "int8", "click_country": "int8",
        "click_region": "int8", "click_referrer_type": "int8",
    }
    article_dtypes = {
        "article_id": "int32", COL_CATEGORY: "int16",
        COL_CREATED: "int64", COL_WORDS: "int16",
    }

    with Timer("Loading data"):
        click_df = pd.read_csv(TRAIN_CLICK_PATH, dtype=click_dtypes)
        article_df = pd.read_csv(ARTICLES_PATH, dtype=article_dtypes)
        try:
            article_emb_df = pd.read_csv(ARTICLES_EMB_PATH)
        except FileNotFoundError:
            article_emb_df = None

    click_df = click_df.drop_duplicates(subset=[COL_USER, COL_ITEM, COL_TIME])  # 这三列如果都相同实际上是重复的点击记录，去重后更干净
    click_df = click_df.sort_values([COL_USER, COL_TIME]).reset_index(drop=True)

    print(f"Clicks: {len(click_df):,}, Users: {click_df[COL_USER].nunique():,}, "
          f"Articles: {article_df['article_id'].nunique():,}")
    return click_df, article_df, article_emb_df


def leave_one_out_split(click_df):
    """Leave-One-Out split: each user's last click becomes test label.

    For users with only 1 click, that click goes to both train and test.

    Returns:
        train_df: all clicks except each user's last
        test_labels: dict {user_id: true_article_id}
    """
    click_df = click_df.sort_values([COL_USER, COL_TIME])
    last_clicks = click_df.groupby(COL_USER).tail(1)
    test_labels = dict(zip(last_clicks[COL_USER], last_clicks[COL_ITEM]))

    train_df = click_df.drop(last_clicks.index).reset_index(drop=True)

    # Single-click users: add their only click back to train
    # single_click_users = set(test_labels.keys()) - set(train_df[COL_USER].unique())
    # if single_click_users:
    #     single_rows = click_df[click_df[COL_USER].isin(single_click_users)]
    #     train_df = pd.concat([train_df, single_rows], ignore_index=True)
    #     train_df = train_df.sort_values([COL_USER, COL_TIME]).reset_index(drop=True)
    # 
    print(f"LOO split: train={len(train_df):,} clicks, test={len(test_labels):,} users")
    return train_df, test_labels


def build_user_item_time_dict(click_df):
    # 构建用户-物品索引 字典，键是用户ID，值是一个列表，列表中的元素是一个元组，包含物品ID和时间戳。列表按照时间戳排序。
    """Build {user_id: [(item_id, timestamp), ...]} sorted by time."""
    d = defaultdict(list)
    for row in click_df[[COL_USER, COL_ITEM, COL_TIME]].itertuples(index=False):
        d[row[0]].append((row[1], row[2]))
    for uid in d:
        d[uid].sort(key=lambda x: x[1])  # 按时间戳排序
    return dict(d)


def build_item_user_time_dict(click_df):
    # 构建物品-用户索引 字典，键是物品ID，值是一个列表，列表中的元素是一个元组，包含用户ID和时间戳。列表按照时间戳排序。
    """Build {item_id: [(user_id, timestamp), ...]} sorted by time."""
    d = defaultdict(list)
    for row in click_df[[COL_USER, COL_ITEM, COL_TIME]].itertuples(index=False):
        d[row[1]].append((row[0], row[2]))
    for iid in d:
        d[iid].sort(key=lambda x: x[1])
    return dict(d)


def build_item_info_dicts(article_df):
    """Build item info lookup dicts from article metadata.

    Returns:
        item_type_dict: {article_id: category_id}
        item_words_dict: {article_id: words_count}
        item_created_time_dict: {article_id: normalized_created_at_ts}
        item_created_abs_time_dict: {article_id: raw_created_at_ts_ms}
    """
    item_type_dict = dict(zip(article_df["article_id"], article_df[COL_CATEGORY]))
    item_words_dict = dict(zip(article_df["article_id"], article_df[COL_WORDS]))

    created_ts = article_df[COL_CREATED].values.astype(np.float64)
    ts_min, ts_max = created_ts.min(), created_ts.max()
    item_created_time_dict = dict(
        zip(article_df["article_id"], (created_ts - ts_min) / (ts_max - ts_min + 1e-8))
    )  # 归一化物品创建时间特征 为什么要归一化？因为原始的时间戳数值很大，直接使用可能会导致模型训练不稳定，归一化后数值范围在0-1之间，更适合模型学习。
    item_created_abs_time_dict = dict(zip(article_df["article_id"], article_df[COL_CREATED]))

    return {
        "item_type_dict": item_type_dict,
        "item_words_dict": item_words_dict,
        "item_created_time_dict": item_created_time_dict,
        "item_created_abs_time_dict": item_created_abs_time_dict,
    }


def build_item_emb_dict(article_emb_df):
    """Build {article_id: np.array(250,)} from embedding CSV, L2-normalized."""
    
    if article_emb_df is None:
        return {}
    emb_cols = [c for c in article_emb_df.columns if c.startswith("emb_")]
    ids = article_emb_df["article_id"].values   # shape: (255755,)
    embs = article_emb_df[emb_cols].values.astype(np.float32)   # shape: (255755, 250)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)  # shape: (255755, 1)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return dict(zip(ids, embs))


def get_topk_popular(click_df, k=50):
    """Return top-k most clicked article_ids."""
    return click_df[COL_ITEM].value_counts().head(k).index.tolist()
