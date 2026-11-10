import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from .metrics import top_k_items


def user_means(R):
    counts = np.diff(R.indptr)
    sums = np.asarray(R.sum(axis=1)).ravel()
    return np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0)


def center_by_user(R, means):
    Rc = R.astype(np.float64).copy()
    Rc.data -= np.repeat(means, np.diff(R.indptr))
    return Rc


class BaseRecommender:
    """Общий интерфейс: score_users возвращает оценки всех книг для набора пользователей."""

    def score_users(self, user_idx):
        raise NotImplementedError

    def get_recommendations(self, user_id, N=5):
        """Топ-N book_id для пользователя (без уже оценённых в train)."""
        u = user_id - 1
        scores = np.asarray(self.score_users(np.array([u])), dtype=np.float64)
        seen = self.R[u].indices
        scores[0, seen] = -np.inf
        return (top_k_items(scores, N)[0] + 1).tolist()


class PopularityRecommender(BaseRecommender):
    """Средний рейтинг книги среди книг, у которых не меньше min_ratings оценок."""

    def __init__(self, min_ratings=50):
        self.min_ratings = min_ratings

    def fit(self, R):
        self.R = R
        counts = np.diff(R.tocsc().indptr)
        means = np.asarray(R.sum(axis=0)).ravel() / np.maximum(counts, 1)
        self.item_scores = np.where(counts >= self.min_ratings, means, -1.0)
        self.counts, self.means = counts, means
        return self

    def top_books(self, N=10):
        order = np.lexsort((-self.counts, -self.item_scores))[:N]
        return pd.DataFrame({
            "book_id": order + 1,
            "mean_rating": self.means[order].round(3),
            "n_ratings": self.counts[order],
        })

    def score_users(self, user_idx):
        # небольшая добавка за число оценок, чтобы разводить равные средние
        base = self.item_scores + 1e-6 * self.counts
        return np.tile(base, (len(user_idx), 1))


class ContentRecommender(BaseRecommender):
    """TF-IDF по профилю книги: название + теги."""

    def __init__(self, top_tags=20, min_df=2):
        self.top_tags = top_tags
        self.min_df = min_df

    @staticmethod
    def build_profiles(books, clean_tags, top_tags=20):
        tags_top = (
            clean_tags.sort_values(["book_id", "count"], ascending=[True, False])
            .groupby("book_id").head(top_tags)
        )
        # тег как одно слово: science-fiction -> science_fiction
        tags_text = (
            tags_top.assign(tok=tags_top["tag_name"].str.replace("-", "_", regex=False))
            .groupby("book_id")["tok"].apply(" ".join)
        )
        title = books["original_title"].fillna(books["title"]).fillna("")
        profiles = pd.Series(title.values, index=books["id"].values)
        profiles = profiles + " " + tags_text.reindex(profiles.index).fillna("")
        return profiles.sort_index()

    def fit(self, books, clean_tags, R=None):
        self.R = R
        self.books = books.set_index("id")
        self.profiles = self.build_profiles(books, clean_tags, self.top_tags)
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=self.min_df,
                                          token_pattern=r"(?u)\b\w[\w_]+\b")
        self.X = self.vectorizer.fit_transform(self.profiles.values)
        return self

    def get_similar_books(self, book_id, N=5):
        i = book_id - 1
        sims = linear_kernel(self.X[i], self.X).ravel()
        sims[i] = -1
        top = np.argsort(-sims)[:N]
        return pd.DataFrame({
            "book_id": top + 1,
            "title": self.books.loc[top + 1, "title"].values,
            "similarity": sims[top].round(3),
        })

    def score_users(self, user_idx, threshold=4):
        # профиль пользователя = сумма TF-IDF векторов книг, которые ему понравились
        liked = (self.R[user_idx] >= threshold).astype(np.float64)
        profile = normalize(liked @ self.X)
        return (profile @ self.X.T).toarray()


class ItemBasedCF(BaseRecommender):
    """Item-based CF: adjusted cosine (центрирование по пользователю), храню top-k соседей."""

    def __init__(self, k=50, shrinkage=25, chunk_size=1000):
        self.k = k
        self.shrinkage = shrinkage
        self.chunk_size = chunk_size

    def fit(self, R):
        self.R = R.tocsr()
        n_items = R.shape[1]
        self.means = user_means(self.R)
        Rc = center_by_user(self.R, self.means)
        norms = np.sqrt(np.asarray(Rc.multiply(Rc).sum(axis=0)).ravel())
        X = (Rc @ diags(1.0 / np.maximum(norms, 1e-12))).tocsc()
        B = (self.R > 0).astype(np.float32).tocsc()
        XT, BT = X.T.tocsr(), B.T.tocsr()

        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.chunk_size):
            chunk = np.arange(start, min(start + self.chunk_size, n_items))
            sim = (XT[chunk] @ X).toarray()
            common = (BT[chunk] @ B).toarray()
            sim *= common / (common + self.shrinkage)  # штраф за малое число общих читателей
            sim[np.arange(len(chunk)), chunk] = 0.0
            sim[sim < 0] = 0.0
            top = np.argpartition(-sim, self.k, axis=1)[:, :self.k]
            top_vals = np.take_along_axis(sim, top, axis=1)
            rows.append(np.repeat(chunk, self.k))
            cols.append(top.ravel())
            vals.append(top_vals.ravel())
        S = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_items, n_items))
        S.eliminate_zeros()
        self.S = S
        self.ST = S.T.tocsr()
        return self

    def predict(self, user_ids, book_ids):
        """Взвешенное среднее оценок пользователя по K соседям книги, которые он оценил."""
        u = np.asarray(user_ids) - 1
        i = np.asarray(book_ids) - 1
        S_rows = self.S[i]
        R_rows = self.R[u]
        num = np.asarray(S_rows.multiply(R_rows).sum(axis=1)).ravel()
        den = np.asarray(S_rows.multiply(R_rows > 0).sum(axis=1)).ravel()
        pred = np.where(den > 0, num / np.maximum(den, 1e-12), self.means[u])
        self.last_coverage = float((den > 0).mean())
        return np.clip(pred, 1, 5)

    def similar_books(self, book_id, N=5):
        row = self.S[book_id - 1]
        order = np.argsort(-row.data)[:N]
        return pd.DataFrame({"book_id": row.indices[order] + 1, "similarity": row.data[order].round(3)})

    def score_users(self, user_idx, like_threshold=4):
        # для топ-N: сумма схожестей с книгами, которые понравились (неявный фидбек, оценка >= 4)
        liked = (self.R[user_idx] >= like_threshold).astype(np.float32)
        return (liked @ self.ST).toarray()


class SVDRecommender(BaseRecommender):
    """SVD через svds: по оценкам (implicit=False) или по "понравилось" >= 4 (implicit=True)."""

    def __init__(self, k=50, implicit=False, like_threshold=4, seed=42):
        self.k = k
        self.implicit = implicit
        self.like_threshold = like_threshold
        self.seed = seed

    def fit(self, R):
        self.R = R.tocsr()
        if self.implicit:
            M = (self.R >= self.like_threshold).astype(np.float64)
            self.means = np.zeros(R.shape[0])
        else:
            self.means = user_means(self.R)
            M = center_by_user(self.R, self.means)
        U, s, Vt = svds(M, k=self.k, random_state=self.seed)
        order = np.argsort(-s)
        self.s, self.Vt = s[order], Vt[order]
        self.P = U[:, order] * self.s
        return self

    def predict(self, user_ids, book_ids):
        u = np.asarray(user_ids) - 1
        i = np.asarray(book_ids) - 1
        pred = self.means[u] + np.einsum("ij,ij->i", self.P[u], self.Vt[:, i].T)
        return pred if self.implicit else np.clip(pred, 1, 5)

    def score_users(self, user_idx):
        return self.means[user_idx, None] + self.P[user_idx] @ self.Vt


def _row_max_norm(scores):
    return scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-12)


class HybridRecommender(BaseRecommender):
    """Нет лайков у пользователя -> популярное, новая книга -> beta * контент, иначе CF + gamma * контент."""

    def __init__(self, cf, content, popularity, beta=0.3, gamma=0.2, like_threshold=4):
        self.cf, self.content, self.popularity = cf, content, popularity
        self.beta, self.gamma = beta, gamma
        self.like_threshold = like_threshold

    def fit(self, R):
        self.R = R.tocsr()
        self.is_new_book = np.diff(self.R.tocsc().indptr) == 0
        return self

    def score_users(self, user_idx):
        cf = _row_max_norm(self.cf.score_users(user_idx))
        cb = _row_max_norm(self.content.score_users(user_idx))
        scores = np.where(self.is_new_book, self.beta * cb, cf + self.gamma * cb)
        n_liked = np.asarray((self.R[user_idx] >= self.like_threshold).sum(axis=1)).ravel()
        cold = n_liked == 0
        if cold.any():
            scores[cold] = self.popularity.score_users(np.flatnonzero(cold))
        return scores
