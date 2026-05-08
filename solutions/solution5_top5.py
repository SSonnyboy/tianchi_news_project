"""Solution 5: top5 approach.

ItemCF (top5_weighted) + Swing + Item2Vec recall
-> LightGBM Ranker (LambdaRank) with dual-axis negative sampling.
"""
import os
import numpy as np
import pandas as pd
from solutions.base_pipeline import BasePipeline
from src.recall.itemcf import ItemCFRecall
from src.recall.swing import SwingRecall
from src.recall.item2vec_recall import Item2VecRecall
from src.recall.combiner import combine_recall_results
from src.negative_sampling import dual_axis_negative_sampling
from src.features import build_ranking_features, get_feature_columns
from src.ranking.lgb_ranker import LGBLambdaRanker
from src.config import CACHE_DIR, COL_USER


class Solution5Top5(BasePipeline):
    name = "S5: top5"
    description = "ItemCF + Swing + Item2Vec -> LightGBM Ranker (LambdaRank)"

    def run_recall(self):
        cache = os.path.join(CACHE_DIR, "s5")
        os.makedirs(cache, exist_ok=True)

        # Channel 1: ItemCF top5_weighted
        print("\n[ItemCF top5_weighted]")
        itemcf = ItemCFRecall(variant="top5_weighted")
        itemcf_sim = itemcf.build_similarity(
            self.user_item_time_dict,
            item_created_time_dict=self.item_info_dicts["item_created_time_dict"],
            save_path=os.path.join(cache, "itemcf_sim.pkl"),
        )
        itemcf_recall = itemcf.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, itemcf_sim,
            topk=50, item_topk_click=self.item_topk_click,
        )

        # Channel 2: Swing
        print("\n[Swing]")
        swing = SwingRecall(alpha=5.0, max_user_hist=8, min_user_hist=3)
        swing_sim = swing.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "swing_sim.pkl"),
        )
        swing_recall = swing.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, swing_sim, topk=50,
        )

        # Channel 3: Item2Vec
        print("\n[Item2Vec]")
        i2v = Item2VecRecall(embedding_dim=64, window=5)
        i2v_sim = i2v.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "item2vec.pkl"),
        )
        i2v_recall = i2v.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, i2v_sim, topk=50,
        )

        return combine_recall_results(
            {"itemcf": itemcf_recall, "swing": swing_recall, "item2vec": i2v_recall},
            weight_dict={"itemcf": 1.5, "swing": 1.0, "item2vec": 1.0},
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
            item_emb_dict=self.item_emb_dict,
        )

        # Dual-axis negative sampling
        feature_df = dual_axis_negative_sampling(
            feature_df, self.test_labels, min_samples=1, max_samples=5,
        )

        feature_cols = get_feature_columns()
        # Add embedding sim if available
        if "emb_sim_last" in feature_df.columns:
            feature_cols.append("emb_sim_last")

        available_cols = [c for c in feature_cols if c in feature_df.columns]
        X = feature_df[available_cols].values
        y = feature_df["label"].values
        user_ids = feature_df[COL_USER].values

        print(f"Features: {len(available_cols)} cols, {len(X)} rows, "
              f"pos rate: {y.mean():.4f}")

        # 80/20 user split
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

        feature_df["pred_score"] = ranker.predict(X)
        return feature_df
