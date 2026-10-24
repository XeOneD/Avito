"""Обучение на всех train-запросах и ответ для benchmark_queries.

    python -m candgen.submit                 # → answer.csv

Конфигурацию (флаги --no-dense и т.д.) выбирайте по результатам candgen.validate.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from candgen import cli, data, features, indexes, pipeline, ranker
from candgen.config import Config

log = logging.getLogger(__name__)
K = 50


def pad(predicted: np.ndarray, fallback: np.ndarray, k: int = K) -> np.ndarray:
    """Дополняет ответ до k объявлений из fallback без повторов."""
    if len(predicted) >= k:
        return predicted[:k]
    extra = fallback[~np.isin(fallback, predicted)][: k - len(predicted)]
    return np.concatenate([predicted, extra])


def check_answer(answer: pd.DataFrame, item_ids: set[str], n_queries: int) -> None:
    assert len(answer) == n_queries and answer["query_id"].is_unique
    for row in answer.itertuples():
        ids = row.answer.split(" ")
        assert len(ids) == K and len(set(ids)) == K, row.query_id
        assert all(re.fullmatch(r"[0-9a-f]{16}", i) and i in item_ids for i in ids), row.query_id


def run(cfg: Config, out: Path = Path("answer.csv"), max_queries: int | None = None) -> pd.DataFrame:
    items = data.load_items(cfg)
    index = indexes.load_or_build(items, cfg)
    queries = data.build_train_queries(cfg, items)
    if max_queries and max_queries < len(queries):
        queries = queries.sample(max_queries, random_state=cfg.seed).reset_index(drop=True)

    cfg_all = Config(**{**cfg.__dict__, "holdout_share": 0.0})
    split = data.split_queries(queries, cfg_all)
    # Для benchmark популярность считается по всем train-запросам.
    popularity = data.item_popularity(queries, index.n_items)
    fit = pipeline.training_frame(split.fit, index, cfg, popularity, counted_in_popularity=True)
    es = pipeline.training_frame(split.early_stop, index, cfg, popularity, counted_in_popularity=True)
    model = ranker.train(fit, es, features.feature_columns(cfg), cfg)

    bench = data.load_benchmark_queries(cfg)
    predicted = pipeline.rank(bench, index, cfg, model, popularity)

    by_popularity = np.argsort(-popularity, kind="stable")
    item_ids = index.items["item_id"].to_numpy()
    rows = []
    for q, query_id in enumerate(bench["query_id"]):
        top = pad(predicted.get(q, np.array([], dtype=int)), by_popularity)
        rows.append({"query_id": query_id, "answer": " ".join(item_ids[top])})
    answer = pd.DataFrame(rows)
    check_answer(answer, set(item_ids), len(bench))
    answer.to_csv(out, index=False)
    padded = sum(len(predicted.get(q, [])) < K for q in range(len(bench)))
    log.info("Сохранено %s: %d запросов, дополнено популярными: %d", out, len(answer), padded)
    return answer


def main() -> None:
    args = cli.parser(__doc__.splitlines()[0]).parse_args()
    run(cli.config_from_args(args), max_queries=args.max_queries)


if __name__ == "__main__":
    main()
