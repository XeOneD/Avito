# Валидация: full

Запросов: fit 18266, early stop 2642, hold-out 5648. Средний размер пула: 1935.

| Метод | Recall@50 |
|---|---|
| full top-50 | 0.3215 |
| title top-50 | 0.2457 |
| char3 top-50 | 0.2429 |
| full_loc top-50 | 0.7646 |
| dense top-50 | 0.2111 |
| пул кандидатов (потолок) | 0.9721 |
| ранкер | 0.8615 |

Вклад признаков (доля gain):

| Признак | Gain |
|---|---|
| rank_full_loc | 0.361 |
| loc_match | 0.260 |
| item_popularity | 0.113 |
| overlap_recall | 0.050 |
| infm_params_recall | 0.031 |
| rank_full | 0.029 |
| dense_cos | 0.020 |
| rank_char3 | 0.015 |
| bm25_full | 0.015 |
| item_desc_len | 0.012 |
| item_reviews_log | 0.011 |
| rank_dense | 0.011 |
| rank_title | 0.011 |
| item_title_len | 0.007 |
| bm25_char3 | 0.007 |
| item_price_log | 0.006 |
| char3_jaccard | 0.006 |
| item_rating | 0.006 |
| overlap_jaccard | 0.006 |
| bm25_title | 0.005 |
| desc_recall | 0.005 |
| n_sources | 0.005 |
| title_recall | 0.003 |
| query_len | 0.003 |
| params_recall | 0.002 |
| cat_match | 0.000 |
