"""Item2Vec recall (from top5).

Similar to Word2Vec recall but with different default parameters
(dim=64, window=5) and uses Faiss for retrieval.
"""
import os
import pickle
import numpy as np
from src.recall.base import BaseRecall


class Item2VecRecall(BaseRecall):
    name = "item2vec"

    def __init__(self, embedding_dim=64, window=5, sg=1):
        self.embedding_dim = embedding_dim
        self.window = window
        self.sg = sg

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, **kwargs):
        """Train Word2Vec on click sequences.

        Returns:
            item_emb_dict: {article_id: np.array(embedding_dim,)}
        """
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached Item2Vec model: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        from gensim.models import Word2Vec

        print("Training Item2Vec model...")
        sentences = [
            [str(iid) for iid, _ in items]
            for items in user_item_time_dict.values()
            if len(items) >= 2
        ]

        model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=1,
            sg=self.sg,
            workers=4,
            epochs=1,
            seed=42,
        )

        item_emb_dict = {}
        for word in model.wv.index_to_key:
            item_emb_dict[int(word)] = model.wv[word].astype(np.float32)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(item_emb_dict, f)
            print(f"Cached Item2Vec model: {save_path}")
        return item_emb_dict

    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        """Use Faiss to find nearest neighbors for user's last item."""
        import faiss

        item_emb_dict = sim
        clicked = {iid for iid, _ in user_items}

        if not user_items:
            return []

        last_iid = user_items[-1][0]
        if last_iid not in item_emb_dict:
            return []

        # Build Faiss index
        all_ids = list(item_emb_dict.keys())
        all_embs = np.array([item_emb_dict[iid] for iid in all_ids])

        # L2 normalize
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_embs = all_embs / norms

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(all_embs)

        query = item_emb_dict[last_iid].reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)

        scores, indices = index.search(query, topk + len(clicked) + 10)

        result = []
        for score, idx in zip(scores[0], indices[0]):
            iid = all_ids[idx]
            if iid not in clicked:
                result.append((iid, float(score)))
            if len(result) >= topk:
                break
        return result
