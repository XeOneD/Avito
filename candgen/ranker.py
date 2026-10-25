"""LightGBM LambdaRank поверх кандидатов.

LambdaRank вместо бинарного классификатора: задача - порядок кандидатов внутри запроса,
и ранжирующая функция потерь оптимизирует именно его. Ранняя остановка - по NDCG@50
на отдельных запросах, которые не входят ни в обучение, ни в hold-out.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from candgen.config import Config


def _groups(frame: pd.DataFrame) -> np.ndarray:
    # LightGBM требует, чтобы строки одного запроса шли подряд.
    q = frame["q"].to_numpy()
    if len(q) and (np.diff(q) < 0).any():
        raise ValueError("Кадр должен быть отсортирован по q")
    return np.unique(q, return_counts=True)[1]


def train(fit: pd.DataFrame, early_stop: pd.DataFrame, features: list[str], cfg: Config) -> lgb.LGBMRanker:
    params = dict(cfg.lgb_params)
    eval_at = params.pop("eval_at", [50])
    model = lgb.LGBMRanker(random_state=cfg.seed, **params)
    model.fit(
        fit[features],
        fit["label"],
        group=_groups(fit),
        eval_set=[(early_stop[features], early_stop["label"])],
        eval_group=[_groups(early_stop)],
        eval_at=eval_at,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )
    return model


def top_items(frame: pd.DataFrame, scores: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """{q: k объявлений с наибольшим скором}."""
    q = frame["q"].to_numpy()
    item = frame["item"].to_numpy()
    order = np.lexsort((item, -scores, q))
    q_sorted = q[order]
    starts = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])
    stops = np.r_[starts[1:], len(order)]
    return {int(q_sorted[s]): item[order[s : min(s + k, e)]] for s, e in zip(starts, stops, strict=True)}
