"""Item2Vec recall: Word2Vec embeddings + FAISS nearest neighbor search.

Treats each user's click history as a "sentence" and each article ID as a
"word", trains Skip-Gram to get item vectors, then uses FAISS for fast
ANN retrieval.
"""
import os
import pickle
from collections import defaultdict

import numpy as np
from gensim.models import Word2Vec

from src.recall.base import BaseRecall


class Item2VecRecall(BaseRecall):
    name = "item2vec"

    def __init__(self, embedding_dim=128, window=3, negative=5, epochs=3):
        self.embedding_dim = embedding_dim
        self.window = window
        self.negative = negative
        self.epochs = epochs

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, **kwargs):
        """Train Word2Vec and build FAISS index.

        Returns:
            dict with keys: "index" (FAISS index), "item_ids" (np array
            mapping internal FAISS index -> article_id)
        """
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached Item2Vec: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        print(f"Building Item2Vec (dim={self.embedding_dim}, "
              f"window={self.window}, neg={self.negative})...")

        # --- Build sentences (each user's click sequence = one sentence) ---
        sentences = []
        for uid, item_time_list in user_item_time_dict.items():
            seq = [str(iid) for iid, _ in item_time_list]
            if len(seq) >= 2:
                sentences.append(seq)

        print(f"  Training Word2Vec on {len(sentences)} sequences...")
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=1,
            sg=1,           # Skip-Gram
            hs=0,           # negative sampling
            negative=self.negative,
            workers=4,
            epochs=self.epochs,
            seed=42,
        )

        # --- Extract vectors into FAISS index ---
        import faiss

        item_ids = []
        vectors = []
        for iid_str, idx in model.wv.key_to_index.items():
            item_ids.append(int(iid_str))
            vectors.append(model.wv[idx])

        item_ids = np.array(item_ids, dtype=np.int64)
        vectors = np.array(vectors, dtype=np.float32)

        # Normalize for cosine similarity (inner product on unit vectors)
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(vectors)

        result = {
            "index": index,
            "item_ids": item_ids,
        }

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"  Cached Item2Vec: {save_path}")

        print(f"  Indexed {len(item_ids)} items.")
        return result

    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        """Recommend using FAISS nearest neighbor search.

        Uses the user's last 2 clicked items as seeds, consistent with
        ItemCF and Swing recall.
        """
        index = sim["index"]
        item_ids = sim["item_ids"]
        id_set = set(item_ids.tolist())
        clicked = {iid for iid, _ in user_items}

        rank = defaultdict(float)
        recent = user_items[-2:][::-1]

        for loc, (seed_iid, _) in enumerate(recent):
            if seed_iid not in id_set:
                continue
            # Find the internal index of seed item
            seed_idx = np.where(item_ids == seed_iid)[0]
            if len(seed_idx) == 0:
                continue
            seed_vec = index.reconstruct(int(seed_idx[0])).reshape(1, -1)

            # Search top-100 nearest neighbors
            k = min(100, index.ntotal)
            distances, indices = index.search(seed_vec, k)

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                cand_iid = int(item_ids[idx])
                if cand_iid in clicked or cand_iid == seed_iid:
                    continue
                # dist is cosine similarity (higher = more similar)
                rank[cand_iid] += float(dist) * (0.7 ** loc)

        result = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:topk]

        # Fallback with popular items if not enough candidates
        if len(result) < topk:
            item_topk_click = kwargs.get("item_topk_click")
            if item_topk_click:
                seen = {iid for iid, _ in result} | clicked
                for iid in item_topk_click:
                    if iid not in seen:
                        result.append((iid, -1.0))
                        if len(result) >= topk:
                            break

        return result
