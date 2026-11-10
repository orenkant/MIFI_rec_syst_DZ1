from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "Dataset"

# служебные полки goodreads, которые не говорят о содержании книги
JUNK_TAG_PATTERN = (
    r"^(to-?read|currently-reading|favou?rites?|fav|owned|own|i-own|books-i-own|owned-books|"
    r"my-books|my-library|library|default|kindle|ebooks?|e-books?|audio|audio-?books?|own-.*|"
    r"audible|to-buy|wish-?list|book-club|re-?read|series|read|books|"
    r"abandoned|dnf|did-not-finish|have|on-hold|unfinished|tbr|maybe|"
    r"\d+|\d+-stars?|read-in-\d+|read-\d+|.*\d{4}.*|.*-books?-i-own|"
    r"favorite-books|all-time-favorites|my-favorites|books-i-have|"
    r"i-have|have-read|already-read|finished|to-be-read)$"
)


def load_data(data_dir=DATA_DIR):
    ratings = pd.read_csv(data_dir / "ratings.csv")
    books = pd.read_csv(data_dir / "books.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    book_tags = pd.read_csv(data_dir / "book_tags.csv")

    # в старой версии датасета есть повторные оценки, оставляю последнюю
    ratings = ratings.drop_duplicates(["user_id", "book_id"], keep="last").reset_index(drop=True)

    # в book_tags книга задана goodreads id, перевожу в id из ratings
    id_map = books.set_index("book_id")["id"]
    book_tags = book_tags[book_tags["count"] > 0].copy()
    book_tags["book_id"] = book_tags["goodreads_book_id"].map(id_map)
    book_tags = (
        book_tags.drop_duplicates(["book_id", "tag_id"])
        .merge(tags, on="tag_id")[["book_id", "tag_id", "tag_name", "count"]]
    )
    return ratings, books, tags, book_tags


def clean_book_tags(book_tags, min_books=20):
    """Убирает служебные теги и совсем редкие."""
    bt = book_tags[~book_tags["tag_name"].str.fullmatch(JUNK_TAG_PATTERN)]
    n_books = bt.groupby("tag_id")["book_id"].transform("nunique")
    return bt[n_books >= min_books].reset_index(drop=True)


def train_test_split_by_user(ratings, test_size=0.2, min_ratings=5, seed=42):
    """У пользователей с min_ratings+ оценками test_size уходит в тест, остальные целиком в train."""
    rng = np.random.default_rng(seed)
    df = ratings.copy()
    df["_rnd"] = rng.random(len(df))
    rank = df.groupby("user_id")["_rnd"].rank(method="first")
    cnt = df.groupby("user_id")["user_id"].transform("size")
    n_test = np.floor(cnt * test_size)
    is_test = (cnt >= min_ratings) & (rank <= n_test)
    df = df.drop(columns="_rnd")
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def to_csr(df, n_users, n_items, value="rating"):
    """Матрица пользователь x книга, индексы = id - 1."""
    return csr_matrix(
        (df[value].astype(np.float32).values, (df["user_id"].values - 1, df["book_id"].values - 1)),
        shape=(n_users, n_items),
    )
