"""Solution 1: news_recommender approach.

ItemCF (weighted) + YouTubeDNN recall, no ranking stage.
"""
import os
from solutions.base_pipeline import BasePipeline
from src.recall.itemcf import ItemCFRecall
from src.recall.youtube_dnn import YouTubeDNNRecall
from src.recall.combiner import combine_recall_results
from src.config import CACHE_DIR


class Solution1NewsRecommender(BasePipeline):
    name = "S1: news_recommender"
    description = "ItemCF (weighted) + YouTubeDNN, recall-only (no ranking)"

    def run_recall(self):
        cache = os.path.join(CACHE_DIR, "s1")
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
            topk=50, item_topk_click=self.item_topk_click,
        )

        # Channel 2: YouTubeDNN
        print("\n[YouTubeDNN]")
        youtube = YouTubeDNNRecall(
            embedding_dim=16, hidden_units=(128, 64),
            neg_ratio=4, epochs=3, batch_size=512,
        )
        youtube_sim = youtube.build_similarity(
            self.user_item_time_dict,
            save_path=os.path.join(cache, "youtube_dnn.pkl"),
        )
        youtube_recall = youtube.batch_recommend(
            self.test_labels.keys(), self.user_item_time_dict, youtube_sim, topk=50,
        )

        # Combine
        return combine_recall_results(
            {"itemcf": itemcf_recall, "youtube": youtube_recall},
            weight_dict={"itemcf": 1.0, "youtube": 1.2},
            topk=50,
        )
