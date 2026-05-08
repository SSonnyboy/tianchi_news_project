"""Solution 2: top2 approach.

ItemCF (loc-weighted) + Bi-network + Word2Vec recall
-> LightGBM Classifier ranking with GroupKFold.
"""
import os
import numpy as np
import pandas as pd
from solutions.base_pipeline import BasePipeline
from src.recall.itemcf import ItemCFRecall
from src.recall.binetwork import BiNetworkRecall
from src.recall.word2vec_recall import Word2VecRecall
from src.recall.combiner import combine_recall_results
from src.negative_sampling import recall_as_negatives
from src.features import build_ranking_features, get_feature_columns
from src.ranking.lgb_classifier import LGBClassifierRanker
from src.config import CACHE_DIR, COL_USER


class Solution2Top2(BasePipeline):
    name = "S2: top2"
    description = "ItemCF + Bi-network + Word2Vec -> LightGBM Classifier (5-fold GroupKFold)"

    def run_recall(self):
        cache = os.path.join(CACHE_DIR, "s2")
        os.makedirs(cache, exist_ok=True)

        # Channel 1: ItemCF weighted
        print("\n[ItemCF weighted]")
        itemcf = ItemCFRecall(variant="weighted")
        itemcf_sim = itemcf.build_similarity(
            self.user_item_time_dict,
            item_created_time_dict=self.item_info_dicts["item_created_time_dict"],
            save_path=os.path.join(cache, "itemcf_sim.pkl"),
        )
        itemcf_recall = itemcf.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, itemcf_sim,
            topk=100, item_topk_click=self.item_topk_click,
        )

        # Channel 2: Bi-network
        print("\n[Bi-network]")
        binetwork = BiNetworkRecall()
        binet_sim = binetwork.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "binetwork_sim.pkl"),
        )
        binet_recall = binetwork.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, binet_sim, topk=100,
        )

        # Channel 3: Word2Vec
        print("\n[Word2Vec + Annoy]")
        w2v = Word2VecRecall(embedding_dim=256, window=3, n_trees=100)
        w2v_sim = w2v.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "w2v_model.pkl"),
        )
        w2v_recall = w2v.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, w2v_sim, topk=100,
        )

        # Save sim dicts for features
        self._sim_dicts = {
            "itemcf": itemcf_sim,
            "binetwork": binet_sim,
        }

        return combine_recall_results(
            {"itemcf": itemcf_recall, "binetwork": binet_recall, "w2v": w2v_recall},
            weight_dict={"itemcf": 1.0, "binetwork": 1.0, "w2v": 0.1},
            topk=50,
        )

    def run_ranking(self, recall_dict):
        """Build features and train LightGBM Classifier."""
        print("\n[Building ranking features]")
        recall_rows = []
        for uid, items in recall_dict.items():
            for iid, score in items:
                recall_rows.append({"user_id": uid, "article_id": iid, "score": score})
        recall_df = pd.DataFrame(recall_rows)

        feature_df = build_ranking_features(
            recall_df, self.train_df, self.article_df,
            self.item_info_dicts, sim_dicts=self._sim_dicts,
        )

        # Label
        feature_df["label"] = 0
        for uid, true_item in self.test_labels.items():
            mask = (feature_df["user_id"] == uid) & (feature_df["article_id"] == true_item)
            feature_df.loc[mask, "label"] = 1

        feature_cols = get_feature_columns()
        # Add similarity feature columns
        for col in feature_df.columns:
            if col.endswith("_sim_last") and col not in feature_cols:
                feature_cols.append(col)

        available_cols = [c for c in feature_cols if c in feature_df.columns]
        X = feature_df[available_cols].values
        y = feature_df["label"].values
        user_ids = feature_df[COL_USER].values

        print(f"Features: {len(available_cols)} cols, {len(X)} rows, "
              f"pos rate: {y.mean():.4f}")

        # Train
        ranker = LGBClassifierRanker(seed=42)
        ranker.train(X, y, user_ids_train=user_ids, n_folds=5)

        # Predict
        feature_df["pred_score"] = ranker.predict(X)
        return feature_df
