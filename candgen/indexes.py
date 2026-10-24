"""Всё, что считается по корпусу объявлений один раз: индексы BM25, бинарные матрицы
слов для признаков пересечения, разбивка по городам и эмбеддинги."""

import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from candgen.bm25 import BM25
from candgen.config import Config
from candgen.text import CharNgramAnalyzer, WordAnalyzer

log = logging.getLogger(__name__)


@dataclass
class ItemIndex:
    items: pd.DataFrame
    full: BM25
    title: BM25
    char3: BM25
    title_bin: sp.csr_matrix
    desc_bin: sp.csr_matrix
    params_bin: sp.csr_matrix
    all_bin: sp.csr_matrix
    char3_bin: sp.csr_matrix
    by_location: dict
    numeric: pd.DataFrame
    dense: np.ndarray | None = None

    @property
    def n_items(self) -> int:
        return len(self.items)


def _binary(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m, dtype=np.float32, copy=True)
    m.data[:] = 1.0
    return m


def _numeric_features(items: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_rating": items["item_rating"].fillna(-1).to_numpy(np.float32),
            "item_reviews_log": np.log1p(items["item_rating_reviews_count"].fillna(0).clip(lower=0)).to_numpy(
                np.float32
            ),
            "item_price_log": np.log1p(items["item_price"].fillna(0).clip(lower=0)).to_numpy(np.float32),
            "item_title_len": items["item_title_raw"].str.len().to_numpy(np.float32),
            "item_desc_len": items["item_description_raw"].str.len().to_numpy(np.float32),
        }
    )


def build(items: pd.DataFrame, cfg: Config) -> ItemIndex:
    t0 = time.perf_counter()
    words = WordAnalyzer(stem=cfg.stem)
    title, desc, params = items["item_title_raw"], items["item_description_raw"], items["item_infm_params_text"]

    # Заголовок повторён трижды: в нём главный сигнал, а описания бывают очень длинными.
    full = BM25(words).fit(title + " " + title + " " + title + " " + desc + " " + params)
    title_bm25 = BM25(words).fit(title)
    char3 = BM25(CharNgramAnalyzer(3)).fit(title)
    log.info("BM25-индексы построены за %.0f с", time.perf_counter() - t0)

    vec = full.vectorizer
    title_bin = _binary(vec.transform(title))
    desc_bin = _binary(vec.transform(desc))
    params_bin = _binary(vec.transform(params))
    all_bin = _binary(title_bin + desc_bin + params_bin)

    locations = items["item_location_id"].to_numpy()
    order = np.argsort(locations, kind="stable")
    unique_locs, starts = np.unique(locations[order], return_index=True)
    by_location = dict(zip(unique_locs.tolist(), np.split(order, starts[1:]), strict=True))

    index = ItemIndex(
        items=items[["item_id", "item_location_id", "item_category_id"]].copy(),
        full=full,
        title=title_bm25,
        char3=char3,
        title_bin=title_bin,
        desc_bin=desc_bin,
        params_bin=params_bin,
        all_bin=all_bin,
        char3_bin=_binary(char3.doc_term),
        by_location=by_location,
        numeric=_numeric_features(items),
    )
    if cfg.use_dense:
        from candgen.dense import encode_passages

        t0 = time.perf_counter()
        index.dense = encode_passages(title.tolist(), cfg)
        log.info("Эмбеддинги заголовков посчитаны за %.0f с", time.perf_counter() - t0)
    return index


def load_or_build(items: pd.DataFrame, cfg: Config) -> ItemIndex:
    path = cfg.artifacts_dir / f"item_index_{'dense' if cfg.use_dense else 'sparse'}.joblib"
    if path.exists():
        index = joblib.load(path)
        if (
            len(index.items) == len(items)
            and (index.items["item_id"].to_numpy() == items["item_id"].to_numpy()).all()
        ):
            log.info("Индекс загружен из %s", path)
            return index
        log.info("Корпус изменился, индекс пересобирается")
    index = build(items, cfg)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)
    return index
