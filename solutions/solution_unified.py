"""Unified solution: 5-channel recall + LightGBM Ranker.

Recall channels:
  1. ItemCF (weighted variant)
  2. Swing
  3. YouTubeDNN dual-tower
  4. Item2Vec (Word2Vec + FAISS)
  5. Hot (global popularity fallback)

Ranking: LightGBM LambdaRanker with dual-axis negative sampling.
"""
import os
import numpy as np
import pandas as pd
from solutions.base_pipeline import BasePipeline
from src.recall.itemcf import ItemCFRecall
from src.recall.swing import SwingRecall
from src.recall.youtube_dnn import YouTubeDNNRecall
from src.recall.item2vec import Item2VecRecall
from src.recall.hot import HotRecall
from src.recall.combiner import combine_recall_results
from src.negative_sampling import dual_axis_negative_sampling
from src.features import build_ranking_features, get_feature_columns
from src.ranking.lgb_ranker import LGBLambdaRanker
from src.config import CACHE_DIR, COL_USER


class SolutionUnified(BasePipeline):
    name = "Unified"
    description = "ItemCF + Swing + YouTubeDNN + Item2Vec + Hot -> LightGBM Ranker"

    # Recall weights (adjustable)
    RECALL_WEIGHTS = {
        "itemcf": 1.0,
        "swing": 1.0,
        "youtube": 1.2,
        "item2vec": 1.0,
        "hot": 0.5,
    }

    def run_recall(self):
        cache = os.path.join(CACHE_DIR, "unified")
        os.makedirs(cache, exist_ok=True)
        user_ids = list(self.test_labels.keys())

        # --- Channel 1: ItemCF (weighted) ---
        print("\n[ItemCF weighted]")
        itemcf = ItemCFRecall(variant="weighted")
        itemcf_sim = itemcf.build_similarity(
            self.user_item_time_dict,
            item_created_time_dict=self.item_info_dicts["item_created_time_dict"],
            save_path=os.path.join(cache, "itemcf_sim.pkl"),
        )
        itemcf_recall = itemcf.batch_recommend(
            user_ids, self.user_item_time_dict, itemcf_sim,
            topk=50, item_topk_click=self.item_topk_click,
        )

        # --- Channel 2: Swing ---
        print("\n[Swing]")
        swing = SwingRecall(alpha=5.0, max_user_hist=8, min_user_hist=3)
        swing_sim = swing.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "swing_sim.pkl"),
        )
        swing_recall = swing.batch_recommend(
            user_ids, self.user_item_time_dict, swing_sim, topk=50,
        )

        # --- Channel 3: YouTubeDNN ---
        print("\n[YouTubeDNN]")
        youtube = YouTubeDNNRecall(embedding_dim=16, hidden_units=(128, 64),
                                   epochs=5, batch_size=512)
        youtube_sim = youtube.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "youtube_dnn.pkl"),
        )
        youtube_recall = youtube.batch_recommend(
            user_ids, self.user_item_time_dict, youtube_sim, topk=50,
        )

        # --- Channel 4: Item2Vec ---
        print("\n[Item2Vec]")
        item2vec = Item2VecRecall(embedding_dim=128, window=3, negative=5, epochs=3)
        item2vec_sim = item2vec.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "item2vec.pkl"),
        )
        item2vec_recall = item2vec.batch_recommend(
            user_ids, self.user_item_time_dict, item2vec_sim, topk=50,
            item_topk_click=self.item_topk_click,
        )

        # --- Channel 5: Hot (global popularity fallback) ---
        print("\n[Hot]")
        hot = HotRecall(lag_hour_min=3, lag_hour_max=27)
        hot_sim = hot.build_similarity(self.user_item_time_dict)
        hot_recall = hot.batch_recommend(
            user_ids, self.user_item_time_dict, hot_sim, topk=10,
            item_created_abs_time_dict=self.item_info_dicts["item_created_abs_time_dict"],
        )

        # --- Fuse all channels ---
        return combine_recall_results(
            {
                "itemcf": itemcf_recall,
                "swing": swing_recall,
                "youtube": youtube_recall,
                "item2vec": item2vec_recall,
                "hot": hot_recall,
            },
            weight_dict=self.RECALL_WEIGHTS,
            topk=50,
        )


#   {
#       user_id: [(item_id, score), ...],   # 每个用户 top-50 候选
#       ...
#   }

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

        feature_df = dual_axis_negative_sampling(
            feature_df, self.test_labels, min_samples=1, max_samples=5,
        )

        feature_cols = get_feature_columns()
        if "emb_sim_last" in feature_df.columns:
            feature_cols.append("emb_sim_last")

        available_cols = [c for c in feature_cols if c in feature_df.columns]
        X = feature_df[available_cols].values
        y = feature_df["label"].values

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

        ranker = LGBLambdaRanker(seed=777)
        ranker.train(X_train, y_train, X_eval, y_eval, group_train, group_eval)

        feature_df["pred_score"] = ranker.predict(X)
        return feature_df
