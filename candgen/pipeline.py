"""Сборка шагов: обучающие кадры и ранжирование запросов по частям."""

import logging
import time

import numpy as np
import pandas as pd

from candgen import candidates, features
from candgen.config import Config
from candgen.indexes import ItemIndex

log = logging.getLogger(__name__)


def training_frame(
    queries: pd.DataFrame, index: ItemIndex, cfg: Config, popularity: np.ndarray, counted_in_popularity: bool
) -> pd.DataFrame:
    """Кандидаты с метками и признаками для обучения. counted_in_popularity=True значит,
    что выборы этих запросов входят в popularity и их нужно вычесть (leave-one-out)."""
    t0 = time.perf_counter()
    frames = []
    for start in range(0, len(queries), cfg.chunk_queries):
        chunk = queries.iloc[start : start + cfg.chunk_queries]
        cand = candidates.generate(chunk, index, cfg, q_offset=start, sample_for_training=True)
        cand = features.add_labels(cand, chunk, q_offset=start)
        cand["is_train_query"] = np.int8(counted_in_popularity)
        frames.append(features.add_features(cand, chunk, index, cfg, popularity, q_offset=start))
    frame = pd.concat(frames, ignore_index=True)
    # Запросы, у которых в пуле нет ни одного позитива, ранжирующей модели ничего не дают.
    has_pos = frame.groupby("q")["label"].transform("max") > 0
    frame = frame[has_pos].sort_values(["q", "item"], kind="stable").reset_index(drop=True)
    log.info(
        "Обучающий кадр: %d запросов, %d пар, %.1f%% позитивов, %.0f с",
        frame["q"].nunique(),
        len(frame),
        100 * frame["label"].mean(),
        time.perf_counter() - t0,
    )
    return frame


def rank(
    queries: pd.DataFrame,
    index: ItemIndex,
    cfg: Config,
    model,
    popularity: np.ndarray,
    k: int = 50,
    collect_baselines: bool = False,
):
    """Возвращает {q: top-k объявлений}. С collect_baselines дополнительно
    {источник: {q: top-k по рангу этого источника}} и {q: весь пул} для потолка полноты."""
    from candgen.ranker import top_items

    feature_cols = features.feature_columns(cfg)
    predicted: dict[int, np.ndarray] = {}
    baselines: dict[str, dict[int, np.ndarray]] = {r: {} for r in cfg.retrievers}
    pools: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()

    for start in range(0, len(queries), cfg.chunk_queries):
        chunk = queries.iloc[start : start + cfg.chunk_queries]
        cand = candidates.generate(chunk, index, cfg, q_offset=start)
        if cand.empty:
            continue
        cand = features.add_features(cand, chunk, index, cfg, popularity, q_offset=start)
        predicted.update(top_items(cand, model.predict(cand[feature_cols]), k))

        if collect_baselines:
            for r in cfg.retrievers:
                ranks = cand[f"rank_{r}"].to_numpy()
                found = ranks < cfg.missing_rank
                baselines[r].update(top_items(cand[found], -ranks[found].astype(np.float32), k=k))
            for q, group in cand.groupby("q")["item"]:
                pools[int(q)] = group.to_numpy()
        log.info("Ранжировано %d/%d запросов", min(start + cfg.chunk_queries, len(queries)), len(queries))

    log.info("Ранжирование заняло %.0f с", time.perf_counter() - t0)
    if collect_baselines:
        return predicted, baselines, pools
    return predicted
