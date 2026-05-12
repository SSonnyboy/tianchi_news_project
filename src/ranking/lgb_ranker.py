"""LightGBM Ranker with LambdaRank objective (from top3/top5).

Uses listwise ranking loss, optimizing NDCG directly.
"""
import numpy as np
import lightgbm as lgb
from src.ranking.base import BaseRanker

# LightGBM
class LGBLambdaRanker(BaseRanker):
    def __init__(self, num_leaves=31, n_estimators=1000, learning_rate=0.01,
                 subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
                 seed=777):
        self.params = dict(
            objective="lambdarank",
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=seed,
            verbose=-1,
        )
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None,
              group_train=None, group_val=None, **kwargs):
        """Train LambdaRank model.

        Args:
            group_train: list of group sizes per user (for training set)
            group_val: list of group sizes per user (for validation set)
        """
        self.model = lgb.LGBMRanker(**self.params)

        eval_set = None
        eval_group = None
        eval_at = [1, 2, 3, 4, 5]

        if X_val is not None and y_val is not None and group_val is not None:
            eval_set = [(X_val, y_val)]
            eval_group = [group_val]

        self.model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_metric="ndcg",
            eval_at=eval_at,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        return self.model.predict(X_train)

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, num_iteration=self.model.best_iteration_)
