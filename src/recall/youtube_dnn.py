"""YouTubeDNN dual-tower model for recall (from news_recommender).

User tower: user_embedding + mean_pool(hist_embedding) -> DNN -> user_emb
Item tower: item_embedding -> item_emb
Score: dot(user_emb, item_emb)

Uses Faiss for efficient retrieval at inference time.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from src.recall.base import BaseRecall


class YouTubeDNNModel(nn.Module):
    def __init__(self, user_count, item_count, embedding_dim=16,
                 hidden_units=(128, 64), dropout=0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.item_embedding = nn.Embedding(item_count, embedding_dim)
        self.hist_embedding = nn.Embedding(item_count, embedding_dim, padding_idx=0)

        input_dim = embedding_dim * 2  # user_emb + mean(hist_emb)
        layers = []
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        self.user_dnn = nn.Sequential(*layers)

        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        nn.init.normal_(self.hist_embedding.weight, 0, 0.01)

    def forward(self, user_ids, hist_seq, target_items, seq_lens):
        user_emb = self.user_embedding(user_ids)  # (B, D)
        hist_emb = self.hist_embedding(hist_seq)   # (B, L, D)

        # Masked mean pooling
        mask = torch.arange(hist_seq.size(1), device=hist_seq.device).unsqueeze(0)
        mask = mask < seq_lens.unsqueeze(1)  # (B, L)
        mask = mask.unsqueeze(2).float()     # (B, L, 1)
        hist_mean = (hist_emb * mask).sum(1) / (mask.sum(1).clamp(min=1))

        user_input = torch.cat([user_emb, hist_mean], dim=1)
        user_out = self.user_dnn(user_input)  # (B, D)

        item_emb = self.item_embedding(target_items)  # (B, D)
        scores = (user_out * item_emb).sum(dim=1)     # (B,)
        return scores

    def get_user_embedding(self, user_ids, hist_seq, seq_lens):
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            hist_emb = self.hist_embedding(hist_seq)
            mask = torch.arange(hist_seq.size(1), device=hist_seq.device).unsqueeze(0)
            mask = mask < seq_lens.unsqueeze(1)
            mask = mask.unsqueeze(2).float()
            hist_mean = (hist_emb * mask).sum(1) / (mask.sum(1).clamp(min=1))
            user_input = torch.cat([user_emb, hist_mean], dim=1)
            return self.user_dnn(user_input).cpu().numpy()

    def get_item_embedding(self):
        with torch.no_grad():
            return self.item_embedding.weight.cpu().numpy()


class YouTubeDNNDataset(Dataset):
    def __init__(self, samples, max_len=30):
        self.max_len = max_len
        self.users = [s[0] for s in samples]
        self.hist_seqs = [s[1] for s in samples]
        self.targets = [s[2] for s in samples]
        self.labels = [s[3] for s in samples]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        hist = self.hist_seqs[idx][-self.max_len:]
        seq_len = min(len(self.hist_seqs[idx]), self.max_len)
        padded = hist + [0] * (self.max_len - len(hist))
        return {
            "user_id": self.users[idx],
            "hist_seq": padded,
            "target": self.targets[idx],
            "seq_len": seq_len,
            "label": self.labels[idx],
        }


def gen_data_set(user_item_time_dict, neg_ratio=4, seed=42):
    """Generate train/test samples with negative sampling.

    Returns:
        train_samples: [(user_enc, hist_items, target_item, label), ...]
        test_samples: [(user_enc, hist_items, target_item, label), ...]
    """
    rng = np.random.RandomState(seed)
    all_items = set()
    for items in user_item_time_dict.values():
        for iid, _ in items:
            all_items.add(iid)
    all_items = list(all_items)

    train_samples, test_samples = [], []

    for uid, item_time_list in user_item_time_dict.items():
        item_list = [iid for iid, _ in item_time_list]
        if len(item_list) < 2:
            # Single item: use for both train and test
            pos = item_list[0]
            neg_items = rng.choice(all_items, size=neg_ratio, replace=False)
            train_samples.append((uid, [pos], pos, 1))
            for neg in neg_items:
                train_samples.append((uid, [pos], neg, 0))
            test_samples.append((uid, [pos], pos, 1))
            continue

        # Last item for test
        test_item = item_list[-1]
        test_hist = item_list[:-1]
        test_samples.append((uid, test_hist, test_item, 1))

        # Sliding window for training
        for i in range(1, len(item_list)):
            hist = item_list[:i]
            target = item_list[i]
            train_samples.append((uid, hist, target, 1))
            # Negative samples
            neg_items = rng.choice(all_items, size=neg_ratio, replace=False)
            for neg in neg_items:
                train_samples.append((uid, hist, neg, 0))

    return train_samples, test_samples


class YouTubeDNNRecall(BaseRecall):
    name = "youtube_dnn"

    def __init__(self, embedding_dim=16, hidden_units=(128, 64),
                 neg_ratio=4, epochs=5, batch_size=512, lr=0.001):
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.neg_ratio = neg_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def build_similarity(self, user_item_time_dict, save_path=None,
                         use_cache=True, device=None, **kwargs):
        """Train YouTubeDNN and extract embeddings.

        Returns:
            (model, user_enc, item_enc, user_emb, item_emb)
        """
        if use_cache and save_path and os.path.exists(save_path):
            print(f"Loading cached YouTubeDNN: {save_path}")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encode IDs
        user_enc = LabelEncoder()
        item_enc = LabelEncoder()
        all_users = list(user_item_time_dict.keys())
        all_items = set()
        for items in user_item_time_dict.values():
            for iid, _ in items:
                all_items.add(iid)
        all_items = list(all_items)

        user_enc.fit(all_users)
        item_enc.fit(all_items)

        # Remap to encoded IDs (+1 for padding)
        encoded_uit = {}
        for uid, items in user_item_time_dict.items():
            enc_uid = user_enc.transform([uid])[0] + 1
            enc_items = [(item_enc.transform([iid])[0] + 1, ts) for iid, ts in items]
            encoded_uit[enc_uid] = enc_items

        n_users = len(all_users) + 2
        n_items = len(all_items) + 2

        print(f"Training YouTubeDNN: {n_users} users, {n_items} items")
        train_samples, test_samples = gen_data_set(encoded_uit, self.neg_ratio)

        train_ds = YouTubeDNNDataset(train_samples)
        test_ds = YouTubeDNNDataset(test_samples)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        model = YouTubeDNNModel(
            n_users, n_items, self.embedding_dim, self.hidden_units
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            total_loss, n_batch = 0, 0
            for batch in train_loader:
                user_ids = batch["user_id"].to(device)
                hist_seq = torch.tensor(batch["hist_seq"]).to(device)
                targets = batch["target"].to(device)
                seq_lens = batch["seq_len"].to(device)
                labels = batch["label"].float().to(device)

                scores = model(user_ids, hist_seq, targets, seq_lens)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            avg_loss = total_loss / n_batch

            # Validate
            model.eval()
            val_loss, val_n = 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    user_ids = batch["user_id"].to(device)
                    hist_seq = torch.tensor(batch["hist_seq"]).to(device)
                    targets = batch["target"].to(device)
                    seq_lens = batch["seq_len"].to(device)
                    labels = batch["label"].float().to(device)
                    scores = model(user_ids, hist_seq, targets, seq_lens)
                    val_loss += criterion(scores, labels).item()
                    val_n += 1
            val_loss /= val_n

            print(f"Epoch {epoch+1}/{self.epochs} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        # Extract embeddings
        model.eval()
        user_emb = model.get_user_embedding(
            torch.arange(1, n_users).to(device),
            torch.zeros(n_users - 1, 30, dtype=torch.long).to(device),
            torch.zeros(n_users - 1, dtype=torch.long).to(device),
        )
        item_emb = model.get_item_embedding()

        result = (model, user_enc, item_enc, user_emb, item_emb)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Cached YouTubeDNN: {save_path}")
        return result

    def recommend(self, user_id, user_items, sim, topk=50, **kwargs):
        """Use Faiss to find top-k items for user embedding."""
        import faiss

        model, user_enc, item_enc, user_emb, item_emb = sim
        clicked = {iid for iid, _ in user_items}

        try:
            enc_uid = user_enc.transform([user_id])[0]
        except ValueError:
            return []

        u_emb = user_emb[enc_uid].reshape(1, -1)
        norms = np.linalg.norm(u_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        u_emb = u_emb / norms

        # Item embeddings (skip padding at index 0)
        items = item_emb[1:]
        item_ids = item_enc.inverse_transform(np.arange(len(items)))

        norms = np.linalg.norm(items, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        items = items / norms

        index = faiss.IndexFlatIP(items.shape[1])
        index.add(items.astype(np.float32))

        scores, indices = index.search(u_emb.astype(np.float32), topk + len(clicked) + 10)

        result = []
        for score, idx in zip(scores[0], indices[0]):
            iid = int(item_ids[idx])
            if iid not in clicked:
                result.append((iid, float(score)))
            if len(result) >= topk:
                break
        return result
