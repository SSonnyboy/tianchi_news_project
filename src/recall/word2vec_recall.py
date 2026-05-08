"""Word2Vec + Annoy recall (from top2).

Trains Word2Vec on user click sequences, builds Annoy index for fast ANN retrieval.
"""
import os
import pickle
from collections import defaultdict
from src.recall.base import BaseRecall


class Word2VecRecall(BaseRecall):
    name = "word2vec"

    def __init__(self, embedding_dim=256, window=3, sg=1, n_trees=100):
        self.embedding_dim = embedding_dim
        self.window = window
        self.sg = sg
        self.n_trees = n_trees

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, **kwargs):
        """Train Word2Vec and build Annoy index.

        Returns:
            (article_vec_map, annoy_index)
        """
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached W2V model: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        from gensim.models import Word2Vec
        from annoy import AnnoyIndex

        print("Training Word2Vec model...")
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
            hs=0,
            negative=5,
            workers=4,
            epochs=1,
            seed=42,
        )

        print("Building Annoy index...")
        article_vec_map = {}
        index = AnnoyIndex(self.embedding_dim, "angular")
        index.set_seed(42)

        for word in model.wv.index_to_key:
            article_id = int(word)
            vec = model.wv[word]
            article_vec_map[article_id] = vec
            index.add_item(article_id, vec)

        index.build(self.n_trees)
        result = (article_vec_map, index)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Cached W2V model: {save_path}")
        return result

    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        """Use last 1 item for Annoy nearest neighbor retrieval."""
        article_vec_map, annoy_index = sim
        clicked = {iid for iid, _ in user_items}

        if not user_items:
            return []

        last_iid = user_items[-1][0]
        if last_iid not in article_vec_map:
            return []

        ids, distances = annoy_index.get_nns_by_item(
            last_iid, topk + len(clicked) + 10, include_distances=True
        )

        result = []
        for iid, dist in zip(ids, distances):
            if iid not in clicked:
                score = 2.0 - dist  # angular distance to similarity
                result.append((iid, score))
            if len(result) >= topk:
                break
        return result
