"""Feature engineering for the ranking stage.

Union of features from top2, top3, and top5 solutions.
"""
import numpy as np
import pandas as pd
from src.config import COL_USER, COL_ITEM, COL_TIME, COL_CATEGORY, COL_WORDS, COL_CREATED


def build_ranking_features(recall_df, train_df, article_df, item_info_dicts,
                           sim_dicts=None, item_emb_dict=None):
    """Build features for each (user_id, article_id) recall candidate.

    Args:
        recall_df: DataFrame [user_id, article_id, score]
        train_df: training click log
        article_df: article metadata
        item_info_dicts: from data_loader.build_item_info_dicts()
        sim_dicts: dict of {name: i2i_sim} for similarity features
        item_emb_dict: {article_id: emb_vector} for embedding features

    Returns:
        feature_df: DataFrame with all features
    """
    df = recall_df.copy()

    item_words = item_info_dicts["item_words_dict"]
    item_created = item_info_dicts["item_created_abs_time_dict"]

    # === User history aggregation ===
    user_hist = (
        train_df.groupby(COL_USER)
        .agg(
            user_click_count=(COL_ITEM, "count"),
            user_avg_words=(COL_WORDS, "mean"),
            user_last_item=(COL_ITEM, "last"),
            user_last_time=(COL_TIME, "last"),
        )
        .reset_index()
    )

    # User click interval
    sorted_df = train_df.sort_values([COL_USER, COL_TIME])
    sorted_df["click_diff"] = sorted_df.groupby(COL_USER)[COL_TIME].diff()
    user_interval = sorted_df.groupby(COL_USER)["click_diff"].mean().reset_index()
    user_interval.columns = [COL_USER, "user_click_diff_mean"]
    user_hist = user_hist.merge(user_interval, on=COL_USER, how="left")

    # User hour std
    train_df_copy = train_df.copy()
    train_df_copy["click_hour"] = pd.to_datetime(train_df_copy[COL_TIME], unit="ms").dt.hour
    user_hour_std = train_df_copy.groupby(COL_USER)["click_hour"].std().reset_index()
    user_hour_std.columns = [COL_USER, "user_hour_std"]
    user_hist = user_hist.merge(user_hour_std, on=COL_USER, how="left")

    # Last item info
    user_hist["user_last_words"] = user_hist["user_last_item"].map(item_words).fillna(0)
    user_hist["user_last_created"] = user_hist["user_last_item"].map(item_created).fillna(0)

    df = df.merge(user_hist, on=COL_USER, how="left")

    # === Item stats ===
    item_stats = train_df.groupby(COL_ITEM).agg(
        item_click_count=(COL_USER, "count"),
        item_user_count=(COL_USER, "nunique"),
    ).reset_index()
    item_stats.rename(columns={COL_ITEM: "article_id"}, inplace=True)
    df = df.merge(item_stats, on="article_id", how="left")

    # === Article features ===
    df = df.merge(article_df[["article_id", COL_CATEGORY, COL_WORDS, COL_CREATED]],
                  on="article_id", how="left")

    # === Time features ===
    df["candidate_created_diff"] = df[COL_CREATED] - df["user_last_created"]
    df["candidate_click_time_diff"] = df[COL_CREATED] - df["user_last_time"]

    # === Word count features ===
    df["word_diff_last"] = (df[COL_WORDS] - df["user_last_words"]).abs()
    df["word_diff_avg"] = (df[COL_WORDS] - df["user_avg_words"]).abs()

    # === Context features ===
    df["hour"] = pd.to_datetime(df["user_last_time"], unit="ms").dt.hour
    df["weekday"] = pd.to_datetime(df["user_last_time"], unit="ms").dt.weekday
    df["freshness"] = df["user_last_time"] - df[COL_CREATED]

    # === Cross features: user x category ===
    train_with_cat = train_df.merge(
        article_df[["article_id", COL_CATEGORY]], left_on=COL_ITEM, right_on="article_id", how="left"
    )
    user_cat_count = (
        train_with_cat.groupby([COL_USER, COL_CATEGORY])
        .size()
        .reset_index(name="user_category_count")
    )
    df = df.merge(
        user_cat_count.rename(columns={COL_CATEGORY: COL_CATEGORY}),
        on=[COL_USER, COL_CATEGORY], how="left"
    )

    # === Similarity features ===
    if sim_dicts:
        for sim_name, sim in sim_dicts.items():
            col_name = f"{sim_name}_sim_last"
            df[col_name] = df.apply(
                lambda row: sim.get(int(row["user_last_item"]), {}).get(int(row["article_id"]), 0),
                axis=1,
            )

    # === Embedding features ===
    if item_emb_dict:
        def emb_sim(row):
            emb1 = item_emb_dict.get(int(row["user_last_item"]))
            emb2 = item_emb_dict.get(int(row["article_id"]))
            if emb1 is not None and emb2 is not None:
                return float(np.dot(emb1, emb2))
            return 0.0
        df["emb_sim_last"] = df.apply(emb_sim, axis=1)

    # Fill NaN
    df = df.fillna(0)

    return df


def get_feature_columns():
    """Return feature column names for LightGBM."""
    return [
        "score", "user_click_count", "user_avg_words", "user_click_diff_mean",
        "user_hour_std", "user_last_words", "item_click_count", "item_user_count",
        COL_WORDS, COL_CREATED, COL_CATEGORY,
        "candidate_created_diff", "candidate_click_time_diff",
        "word_diff_last", "word_diff_avg",
        "hour", "weekday", "freshness", "user_category_count",
    ]
