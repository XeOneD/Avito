"""Признаки пары запрос-кандидат. Одна функция для обучения, валидации и benchmark -
так признаки на инференсе гарантированно считаются так же, как при обучении."""

import numpy as np
import pandas as pd

from candgen.bm25 import rowwise_dot
from candgen.config import Config
from candgen.data import query_text
from candgen.indexes import ItemIndex


def feature_columns(cfg: Config) -> list[str]:
    cols = [f"rank_{r}" for r in cfg.retrievers]
    cols += ["n_sources", "bm25_full", "bm25_title", "bm25_char3"]
    if cfg.use_dense:
        cols.append("dense_cos")
    cols += [
        "query_len",
        "overlap_recall",
        "overlap_jaccard",
        "title_recall",
        "desc_recall",
        "params_recall",
        "char3_jaccard",
        "infm_params_recall",
        "loc_match",
        "cat_match",
        "item_rating",
        "item_reviews_log",
        "item_price_log",
        "item_title_len",
        "item_desc_len",
    ]
    if cfg.use_popularity:
        cols.append("item_popularity")
    return cols


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b > 0)


def add_features(
    cand: pd.DataFrame,
    queries: pd.DataFrame,
    index: ItemIndex,
    cfg: Config,
    popularity: np.ndarray | None = None,
    q_offset: int = 0,
) -> pd.DataFrame:
    """cand - результат candidates.generate для тех же queries и q_offset.

    popularity - сколько обучающих запросов выбрали объявление. Если в cand есть
    столбец label (обучение), свой собственный выбор вычитается (leave-one-out):
    иначе у позитивов популярность всегда ≥ 1 и признак выдаёт метку.
    """
    q = cand["q"].to_numpy() - q_offset
    item = cand["item"].to_numpy()

    words = index.full.encode(query_text(queries, cfg))
    infm = index.full.encode(queries["search_infm_params_text"].tolist())
    char3 = index.char3.encode(queries["search_query"].tolist())

    q_len = np.diff(words.indptr).astype(np.float32)[q]
    infm_len = np.diff(infm.indptr).astype(np.float32)[q]
    char3_len = np.diff(char3.indptr).astype(np.float32)[q]
    all_len = np.diff(index.all_bin.indptr).astype(np.float32)[item]
    item_char3_len = np.diff(index.char3_bin.indptr).astype(np.float32)[item]

    inter_all = rowwise_dot(words, index.all_bin, q, item)
    inter_char3 = rowwise_dot(char3, index.char3_bin, q, item)

    out = cand.copy()
    rank_cols = [f"rank_{r}" for r in cfg.retrievers]
    out["n_sources"] = (out[rank_cols].to_numpy() < cfg.missing_rank).sum(axis=1).astype(np.float32)
    out["query_len"] = q_len
    out["overlap_recall"] = _safe_div(inter_all, q_len)
    out["overlap_jaccard"] = _safe_div(inter_all, q_len + all_len - inter_all)
    out["title_recall"] = _safe_div(rowwise_dot(words, index.title_bin, q, item), q_len)
    out["desc_recall"] = _safe_div(rowwise_dot(words, index.desc_bin, q, item), q_len)
    out["params_recall"] = _safe_div(rowwise_dot(words, index.params_bin, q, item), q_len)
    out["char3_jaccard"] = _safe_div(inter_char3, char3_len + item_char3_len - inter_char3)
    out["infm_params_recall"] = _safe_div(rowwise_dot(infm, index.params_bin, q, item), infm_len)

    item_loc = index.items["item_location_id"].to_numpy()[item]
    item_cat = index.items["item_category_id"].to_numpy()[item]
    out["loc_match"] = (queries["search_location_id"].to_numpy()[q] == item_loc).astype(np.float32)
    out["cat_match"] = (queries["search_category"].to_numpy()[q] == item_cat).astype(np.float32)

    numeric = index.numeric.iloc[item].reset_index(drop=True)
    for col in numeric.columns:
        out[col] = numeric[col].to_numpy()

    if cfg.use_popularity:
        pop = np.zeros(len(out), np.float32) if popularity is None else popularity[item].copy()
        if "label" in out and "is_train_query" in out:
            pop -= (out["label"].to_numpy() * out["is_train_query"].to_numpy()).astype(np.float32)
        out["item_popularity"] = np.log1p(np.maximum(pop, 0))
    return out


def add_labels(cand: pd.DataFrame, queries: pd.DataFrame, q_offset: int = 0) -> pd.DataFrame:
    pairs = pd.DataFrame(
        {
            "q": np.repeat(np.arange(len(queries)) + q_offset, queries["relevant"].map(len).to_numpy()),
            "item": np.concatenate(queries["relevant"].to_numpy()) if len(queries) else np.array([], int),
        }
    ).astype(np.int32)
    pairs["label"] = 1
    out = cand.merge(pairs, on=["q", "item"], how="left")
    out["label"] = out["label"].fillna(0).astype(np.int8)
    return out
