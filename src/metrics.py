import numpy as np
from scipy.sparse import csr_matrix


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# для одного пользователя, ими проверяю evaluate_topn

def precision_at_k(recommended, relevant, k):
    rec_k = list(recommended)[:k]
    return len(set(rec_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    rec_k = list(recommended)[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    gains = np.array([1.0 if item in relevant else 0.0 for item in list(recommended)[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_len = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_len + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def top_k_items(scores, k):
    """Индексы k лучших по убыванию для каждой строки."""
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def evaluate_topn(model, train_R, test_df, ks=(5, 10), threshold=4, users=None, batch_size=1000):
    """Precision/Recall/nDCG@K: релевантно >= threshold, уже оценённое в train исключаю."""
    n_users, n_items = train_R.shape
    rel_df = test_df[test_df["rating"] >= threshold]
    rel = csr_matrix(
        (np.ones(len(rel_df), dtype=bool), (rel_df["user_id"].values - 1, rel_df["book_id"].values - 1)),
        shape=(n_users, n_items),
    )
    if users is None:
        users = np.unique(rel_df["user_id"].values - 1)
    k_max = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, k_max + 2))
    ideal_cum = np.concatenate([[0.0], np.cumsum(discounts)])

    sums = {f"{m}@{k}": 0.0 for k in ks for m in ("Precision", "Recall", "nDCG")}
    recommended_items = {k: set() for k in ks}
    for start in range(0, len(users), batch_size):
        idx = users[start:start + batch_size]
        scores = np.asarray(model.score_users(idx), dtype=np.float64)
        seen = train_R[idx].tocoo()
        scores[seen.row, seen.col] = -np.inf
        top = top_k_items(scores, k_max)

        rel_batch = rel[idx]
        n_rel = np.asarray(rel_batch.sum(axis=1)).ravel()
        hits = rel_batch.toarray()[np.arange(len(idx))[:, None], top].astype(float)
        for k in ks:
            h = hits[:, :k]
            sums[f"Precision@{k}"] += (h.sum(1) / k).sum()
            sums[f"Recall@{k}"] += (h.sum(1) / np.maximum(n_rel, 1)).sum()
            idcg = ideal_cum[np.minimum(n_rel, k)]
            sums[f"nDCG@{k}"] += ((h * discounts[:k]).sum(1) / np.maximum(idcg, 1e-12)).sum()
            recommended_items[k].update(np.unique(top[:, :k]).tolist())

    result = {name: value / len(users) for name, value in sums.items()}
    for k in ks:
        result[f"Coverage@{k}"] = len(recommended_items[k]) / n_items
    return result
