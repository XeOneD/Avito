"""Генерация кандидатов.

Каждый источник (BM25 по полному тексту, по заголовку, по символьным 3-граммам,
BM25 внутри города запроса и плотный поиск) даёт свой топ. Кандидаты - их объединение.
Для каждого кандидата сохраняются ранги во всех источниках и точные скоры BM25 и e5:
скоры берутся из той же матрицы, по которой строился топ, поэтому они есть у любого
кандидата, а не только у найденных конкретным индексом.
"""

import numpy as np
import pandas as pd

from candgen.bm25 import top_k
from candgen.config import Config
from candgen.data import query_text
from candgen.indexes import ItemIndex


def _ranks_in_union(union: np.ndarray, found: np.ndarray, missing: int) -> np.ndarray:
    """union отсортирован. Для каждого элемента union - его позиция в found или missing."""
    ranks = np.full(len(union), missing, dtype=np.int32)
    if len(found):
        pos = np.searchsorted(union, found)
        ranks[pos] = np.arange(len(found), dtype=np.int32)
    return ranks


def _sample_for_training(
    union: np.ndarray, best_rank: np.ndarray, positives: np.ndarray, cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    """Позитивы из пула + n_hard лучших по рангу негативов + n_random случайных.
    Позитивы вне пула не добавляются: модель на инференсе никогда их не увидит,
    а их признаки (не найден ни одним источником) учили бы её неправильному."""
    is_pos = np.isin(union, positives)
    neg = np.flatnonzero(~is_pos)
    hard = neg[np.argsort(best_rank[neg], kind="stable")[: cfg.n_hard_negatives]]
    rest = np.setdiff1d(neg, hard, assume_unique=True)
    random = rng.choice(rest, size=min(cfg.n_random_negatives, len(rest)), replace=False)
    keep = np.concatenate([np.flatnonzero(is_pos), hard, random])
    return np.sort(keep)


def generate(
    queries: pd.DataFrame,
    index: ItemIndex,
    cfg: Config,
    q_offset: int = 0,
    sample_for_training: bool = False,
) -> pd.DataFrame:
    """Кандидаты для queries. Столбец q - номер запроса (позиция в queries + q_offset)."""
    texts = query_text(queries, cfg)
    q_full = index.full.encode(texts)
    q_title = index.title.encode(texts)
    q_char3 = index.char3.encode(queries["search_query"].tolist())
    q_dense = None
    if cfg.use_dense:
        from candgen.dense import encode_queries

        q_dense = encode_queries(queries["search_query"].tolist(), cfg)

    locations = queries["search_location_id"].to_numpy()
    relevant = queries["relevant"].to_numpy() if sample_for_training else None
    rng = np.random.default_rng(cfg.seed + q_offset)
    missing = cfg.missing_rank
    parts = []

    for start in range(0, len(queries), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(queries))
        s_full = index.full.score_all(q_full[start:stop])
        s_title = index.title.score_all(q_title[start:stop])
        s_char3 = index.char3.score_all(q_char3[start:stop])
        s_dense = q_dense[start:stop] @ index.dense.T if q_dense is not None else None

        for j, qi in enumerate(range(start, stop)):
            found = {
                "full": top_k(s_full[j], cfg.k_full),
                "title": top_k(s_title[j], cfg.k_title),
                "char3": top_k(s_char3[j], cfg.k_char3),
            }
            if cfg.use_location:
                in_city = index.by_location.get(locations[qi])
                found["full_loc"] = (
                    in_city[top_k(s_full[j][in_city], cfg.k_full_loc)]
                    if in_city is not None
                    else np.array([], int)
                )
            if s_dense is not None:
                found["dense"] = top_k(s_dense[j], cfg.k_dense)

            union = np.unique(np.concatenate(list(found.values())))
            if len(union) == 0:
                continue
            ranks = {name: _ranks_in_union(union, ids, missing) for name, ids in found.items()}

            keep = slice(None)
            if sample_for_training:
                best_rank = np.min(np.vstack(list(ranks.values())), axis=0)
                keep = _sample_for_training(union, best_rank, relevant[qi], cfg, rng)

            items = union[keep]
            part = {"q": np.full(len(items), qi + q_offset, dtype=np.int32), "item": items.astype(np.int32)}
            for name, r in ranks.items():
                part[f"rank_{name}"] = r[keep]
            part["bm25_full"] = s_full[j, items]
            part["bm25_title"] = s_title[j, items]
            part["bm25_char3"] = s_char3[j, items]
            if s_dense is not None:
                part["dense_cos"] = s_dense[j, items]
            parts.append(pd.DataFrame(part))

    if not parts:
        return pd.DataFrame(columns=["q", "item"])
    return pd.concat(parts, ignore_index=True)
