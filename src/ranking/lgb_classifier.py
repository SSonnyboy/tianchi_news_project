"""LightGBM Classifier ranker (from top2).

Uses binary classification with GroupKFold cross-validation.
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from src.ranking.base import BaseRanker


class LGBClassifierRanker(BaseRanker):
    def __init__(self, num_leaves=64, max_depth=10, learning_rate=0.05,
                 n_estimators=10000, subsample=0.8, feature_fraction=0.8,
                 reg_alpha=0.5, reg_lambda=0.5, seed=42):
        self.params = dict(
            num_leaves=num_leaves, max_depth=max_depth,
            learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, feature_fraction=feature_fraction,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=seed,
        )
        self.models = []

    def train(self, X_train, y_train, X_val=None, y_val=None,
              user_ids_train=None, n_folds=5, **kwargs):
        """Train with GroupKFold CV grouped by user_id.

        If user_ids_train is provided, uses GroupKFold.
        Otherwise trains a single model.
        """
        if user_ids_train is not None and n_folds > 1:
            gkf = GroupKFold(n_splits=n_folds)
            oof_pred = np.zeros(len(X_train))

            for fold, (trn_idx, val_idx) in enumerate(
                gkf.split(X_train, y_train, groups=user_ids_train)
            ):
                print(f"Fold {fold+1}/{n_folds}")
                model = lgb.LGBMClassifier(**self.params)
                model.fit(
                    X_train[trn_idx], y_train[trn_idx],
                    eval_set=[(X_train[val_idx], y_train[val_idx])],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                )
                self.models.append(model)
                oof_pred[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]

            return oof_pred
        else:
            model = lgb.LGBMClassifier(**self.params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            self.models.append(model)
            return model.predict_proba(X_train)[:, 1]

    def predict(self, X) -> np.ndarray:
        """Average predictions across all fold models."""
        if not self.models:
            raise ValueError("Model not trained yet")
        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict_proba(X)[:, 1]
        return preds / len(self.models)
