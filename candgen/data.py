"""Загрузка данных, сборка обучающих запросов и честное разбиение."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from candgen.config import Config

QUERY_KEY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def _to_float(series: pd.Series) -> pd.Series:
    # Цены и координаты лежат строками вида "500.000000000000000".
    cleaned = series.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def load_items(cfg: Config) -> pd.DataFrame:
    items = pd.read_parquet(cfg.data_dir / "benchmark_items.parquet")
    items["item_id"] = items["item_id"].astype(str)
    items["item_price"] = _to_float(items["item_price"])
    items["item_rating"] = pd.to_numeric(items["item_rating"], errors="coerce")
    items["item_rating_reviews_count"] = pd.to_numeric(items["item_rating_reviews_count"], errors="coerce")
    for col in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        items[col] = items[col].fillna("").astype(str)
    return items.reset_index(drop=True)


def load_benchmark_queries(cfg: Config) -> pd.DataFrame:
    queries = pd.read_parquet(cfg.data_dir / "benchmark_queries.parquet")
    queries["query_id"] = queries["query_id"].astype(str)
    return _clean_queries(queries)


def _clean_queries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["search_query"] = df["search_query"].fillna("").astype(str)
    df["search_infm_params_text"] = df["search_infm_params_text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def query_text(queries: pd.DataFrame, cfg: Config) -> list[str]:
    """Текст запроса для поиска. Одна функция для train, валидации и benchmark:
    в старой версии они различались, и модель видела на инференсе другие признаки."""
    if cfg.infm_in_query:
        return (queries["search_query"] + " " + queries["search_infm_params_text"]).tolist()
    return queries["search_query"].tolist()


def build_train_queries(cfg: Config, items: pd.DataFrame) -> pd.DataFrame:
    """Один запрос = уникальная комбинация QUERY_KEY_COLS. Позитивы - выбранные объявления,
    которые есть в корпусе кандидатов (остальные недостижимы и только шумят)."""
    train = pd.read_parquet(cfg.data_dir / "train.parquet", columns=QUERY_KEY_COLS + ["item_id"])
    train["item_id"] = train["item_id"].astype(str)
    item_index = pd.Series(np.arange(len(items)), index=items["item_id"])
    train = train[train["item_id"].isin(item_index.index)]
    train["item_idx"] = item_index.loc[train["item_id"]].to_numpy()

    keys = train[QUERY_KEY_COLS].fillna("__NA__").astype(str)
    train["query_key"] = keys.agg("||".join, axis=1)
    grouped = train.groupby("query_key", sort=True)
    groups = grouped[QUERY_KEY_COLS].first()
    groups["relevant"] = grouped["item_idx"].unique().map(np.sort)
    return _clean_queries(groups.reset_index())


@dataclass
class Split:
    fit: pd.DataFrame
    early_stop: pd.DataFrame
    holdout: pd.DataFrame


def split_queries(queries: pd.DataFrame, cfg: Config) -> Split:
    """Разбиение по тексту запроса: одинаковый текст в разных городах целиком попадает
    в одну часть. Hold-out не участвует ни в обучении, ни в ранней остановке."""
    texts = queries["search_query"].str.strip().str.lower()
    unique_texts = texts.drop_duplicates().to_numpy()
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(unique_texts)

    n_holdout = int(len(unique_texts) * cfg.holdout_share)
    n_es = int(len(unique_texts) * cfg.early_stop_share)
    holdout_texts = set(unique_texts[:n_holdout])
    es_texts = set(unique_texts[n_holdout : n_holdout + n_es])

    part = np.where(texts.isin(holdout_texts), "holdout", np.where(texts.isin(es_texts), "es", "fit"))
    return Split(
        fit=queries[part == "fit"].reset_index(drop=True),
        early_stop=queries[part == "es"].reset_index(drop=True),
        holdout=queries[part == "holdout"].reset_index(drop=True),
    )


def item_popularity(queries: pd.DataFrame, n_items: int) -> np.ndarray:
    """Сколько обучающих запросов выбрали объявление."""
    if queries.empty:
        return np.zeros(n_items, dtype=np.float32)
    return np.bincount(np.concatenate(queries["relevant"].to_numpy()), minlength=n_items).astype(np.float32)
