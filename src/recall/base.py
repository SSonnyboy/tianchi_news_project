"""Abstract base class for recall algorithms."""
from abc import ABC, abstractmethod


class BaseRecall(ABC):
    """Base class for all recall algorithms.

    Each recall algorithm implements:
    - build_similarity(): build i2i or u2u similarity matrix
    - recommend(): generate candidates for a single user
    - batch_recommend(): generate candidates for all users (default: loop)
    """
    name: str = "base"

    @abstractmethod
    def build_similarity(self, user_item_time_dict, **kwargs):
        """Build similarity structure (i2i dict, embeddings, etc.)."""
        pass

    @abstractmethod
    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        """Return [(item_id, score), ...] for one user.

        Args:
            user_id: target user
            user_items: list of (item_id, timestamp) from user's history, sorted by time
            sim: similarity structure from build_similarity
            topk: number of candidates to return
        """
        pass

    def batch_recommend(self, user_ids, user_item_time_dict, sim, topk=50, **kwargs):
        """Generate recall candidates for multiple users."""
        result = {}
        for uid in user_ids:
            if uid not in user_item_time_dict:
                continue
            items = user_item_time_dict[uid]
            result[uid] = self.recommend(uid, items, sim, topk, **kwargs)
        return result
