"""Solution 3: top3 approach.

Hot (time-windowed) + 30s-interval i2i recall
-> LightGBM Ranker (LambdaRank) with dual-axis negative sampling.
"""
import os
import numpy as np
import pandas as pd
from solutions.base_pipeline import BasePipeline
from src.recall.hot import HotRecall
from src.recall.i2i_30s import I2I30sRecall
from src.recall.combiner import combine_recall_results
from src.negative_sampling import dual_axis_negative_sampling
from src.features import build_ranking_features, get_feature_columns
from src.ranking.lgb_ranker import LGBLambdaRanker
from src.config import CACHE_DIR, COL_USER


class Solution3Top3(BasePipeline):
    name = "S3: top3"
    description = "Hot + 30s-interval i2i -> LightGBM Ranker (LambdaRank)"

    def run_recall(self):
        cache = os.path.join(CACHE_DIR, "s3")
        os.makedirs(cache, exist_ok=True)

        item_abs_time = self.item_info_dicts["item_created_abs_time_dict"]

        # Channel 1: Hot recall (time-windowed: 3-27h)
        print("\n[Hot recall]")
        hot = HotRecall(lag_hour_min=3, lag_hour_max=27)
        hot_sim = hot.build_similarity(self.user_item_time_dict)
        hot_recall = hot.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, hot_sim,
            topk=10, item_created_abs_time_dict=item_abs_time,
        )

        # Channel 2: 30s-interval i2i
        print("\n[30s-interval i2i]")
        i2i_30s = I2I30sRecall()
        i2i_sim = i2i_30s.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "i2i_30s_sim.pkl"),
        )
        i2i_recall = i2i_30s.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, i2i_sim,
            topk=25, item_created_abs_time_dict=item_abs_time,
        )

        return combine_recall_results(
            {"hot": hot_recall, "i2i_30s": i2i_recall},
            weight_dict={"hot": 1.0, "i2i_30s": 1.0},
            topk=50,
        )

    def run_ranking(self, recall_dict):
        """Build features, dual-axis negative sampling, train LambdaRank."""
        print("\n[Building ranking features]")
        recall_rows = []
        for uid, items in recall_dict.items():
            for iid, score in items:
                recall_rows.append({"user_id": uid, "article_id": iid, "score": score})
        recall_df = pd.DataFrame(recall_rows)

        feature_df = build_ranking_features(
            recall_df, self.train_df, self.article_df, self.item_info_dicts,
        )

        # Dual-axis negative sampling
        feature_df = dual_axis_negative_sampling(
            feature_df, self.test_labels, min_samples=1, max_samples=5,
        )

        feature_cols = get_feature_columns()
        available_cols = [c for c in feature_cols if c in feature_df.columns]
        X = feature_df[available_cols].values
        y = feature_df["label"].values
        user_ids = feature_df[COL_USER].values

        print(f"Features: {len(available_cols)} cols, {len(X)} rows, "
              f"pos rate: {y.mean():.4f}")

        # Build group sizes (samples per user for LambdaRank)
        user_counts = feature_df.groupby(COL_USER).size()
        group_train = user_counts.values.tolist()

        # 80/20 user split for train/eval
        unique_users = feature_df[COL_USER].unique()
        np.random.seed(42)
        np.random.shuffle(unique_users)
        n_train = int(0.8 * len(unique_users))
        train_users = set(unique_users[:n_train])
        eval_users = set(unique_users[n_train:])

        train_mask = feature_df[COL_USER].isin(train_users)
        eval_mask = feature_df[COL_USER].isin(eval_users)

        X_train = feature_df.loc[train_mask, available_cols].values
        y_train = feature_df.loc[train_mask, "label"].values
        X_eval = feature_df.loc[eval_mask, available_cols].values
        y_eval = feature_df.loc[eval_mask, "label"].values

        group_train = feature_df.loc[train_mask].groupby(COL_USER).size().values.tolist()
        group_eval = feature_df.loc[eval_mask].groupby(COL_USER).size().values.tolist()

        # Train
        ranker = LGBLambdaRanker(seed=777)
        ranker.train(X_train, y_train, X_eval, y_eval, group_train, group_eval)

        # Predict on all
        feature_df["pred_score"] = ranker.predict(X)
        return feature_df
