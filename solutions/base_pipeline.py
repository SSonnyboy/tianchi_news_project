"""Base pipeline skeleton for all solutions."""
from src.data_loader import (
    load_raw_data, leave_one_out_split,
    build_user_item_time_dict, build_item_info_dicts,
    build_item_emb_dict, get_topk_popular,
)
from src.metrics import evaluate_recall, evaluate_ranking
from src.utils import Timer


class BasePipeline:
    name: str = "base"
    description: str = ""

    def __init__(self, cache_dir="outputs/cache"):
        self.cache_dir = cache_dir
        self.click_df = None
        self.article_df = None
        self.article_emb_df = None
        self.train_df = None
        self.test_labels = None
        self.user_item_time_dict = None
        self.item_info_dicts = None
        self.item_emb_dict = None
        self.item_topk_click = None

    def load_data(self):
        """Load data and build shared structures."""
        self.click_df, self.article_df, self.article_emb_df = load_raw_data()
        self.train_df, self.test_labels = leave_one_out_split(self.click_df)
        self.user_item_time_dict = build_user_item_time_dict(self.train_df)
        self.item_info_dicts = build_item_info_dicts(self.article_df)
        self.item_emb_dict = build_item_emb_dict(self.article_emb_df)
        self.item_topk_click = get_topk_popular(self.train_df, k=50)

    def run_recall(self):
        """Override in subclass. Returns {user_id: [(item_id, score), ...]}."""
        raise NotImplementedError

    def run_ranking(self, recall_dict):
        """Override in subclass if ranking is needed. Returns DataFrame."""
        return None

    def evaluate(self, recall_dict, ranked_df=None):
        """Evaluate recall and optionally ranking."""
        print(f"\n{'='*50}")
        print(f"  {self.name}: {self.description}")
        print(f"{'='*50}")

        print("\n--- Recall Stage ---")
        recall_metrics = evaluate_recall(recall_dict, self.test_labels)

        ranking_metrics = {}
        if ranked_df is not None:
            print("\n--- Ranking Stage ---")
            ranking_metrics = evaluate_ranking(
                ranked_df["label"].values,
                ranked_df["pred_score"].values,
                ranked_df["user_id"].values,
            )

        return {**recall_metrics, **ranking_metrics}

    def run(self):
        """Full pipeline: load -> recall -> [ranking] -> evaluate."""
        with Timer(f"[{self.name}] Loading data"):
            self.load_data()

        with Timer(f"[{self.name}] Running recall"):
            recall_dict = self.run_recall()

        ranked_df = self.run_ranking(recall_dict)
        metrics = self.evaluate(recall_dict, ranked_df)
        return metrics
