# ДЗ 1. Рекомендательная система для книг (goodbooks-10k)

Сравниваю популярное, TF-IDF, item-based CF, SVD и простой гибрид на одних данных.

## Файлы

- `notebook.ipynb` — EDA, модели, метрики, выводы
- `src/data.py` — загрузка, чистка тегов, разбиение
- `src/models.py` — модели
- `src/metrics.py` — RMSE, Precision@K, Recall@K, nDCG@K
- `Dataset/` — csv из [goodbooks-10k](https://www.kaggle.com/datasets/zygmuntz/goodbooks-10k)

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Что сделано

- Убрал повторные оценки, теги перевёл на id книги и почистил от служебных полок.
- Времени в данных нет, поэтому у каждого пользователя с 5+ оценками 20% случайно ушли в тест. Параметры подбирал на валидации.
- Релевантной считаю оценку от 4.
- Popularity: средний рейтинг при не менее чем 50 оценках.
- Content: TF-IDF по названию и тегам, `get_similar_books(book_id, N=5)`.
- Item-based CF: adjusted cosine, top-k соседей, разреженные матрицы.
- SVD: `scipy.sparse.linalg.svds`, по оценкам для RMSE и `get_recommendations(user_id, N=5)` и на неявном фидбеке для топ-N.
- Гибрид: новым пользователям популярное, новым книгам контент, остальным CF плюс немного контента.

## Результаты на тесте

| Модель | P@10 | R@10 | nDCG@10 | RMSE |
|---|---|---|---|---|
| Hybrid (CF + Content) | 0.084 | 0.240 | 0.208 | — |
| Item-based CF | 0.083 | 0.237 | 0.206 | 0.881 |
| SVD неявный | 0.083 | 0.215 | 0.188 | — |
| Content (TF-IDF) | 0.039 | 0.133 | 0.100 | — |
| SVD по оценкам | 0.014 | 0.025 | 0.025 | 0.858 |
| Popularity | 0.0005 | 0.0015 | 0.0009 | — |

Лучше всех CF и гибрид. SVD по оценкам лучший по RMSE, но в топ-N слабый. Популярное почти не работает, потому что у каждой книги в ratings.csv около 100 оценок.
