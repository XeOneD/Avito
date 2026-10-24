"""Честная оценка: Recall@50 на отложенных запросах, тексты которых модель не видела.

    python -m candgen.validate               # полная конфигурация
    python -m candgen.validate --no-dense    # абляция: без e5
    python -m candgen.validate --no-location # абляция: без кандидатов из города запроса

Результат: reports/validation_<конфигурация>.md и .json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from candgen import cli, data, features, indexes, pipeline, ranker
from candgen.config import Config
from candgen.metrics import recall_at_k

log = logging.getLogger(__name__)


def run(cfg: Config, max_queries: int | None = None) -> dict:
    items = data.load_items(cfg)
    index = indexes.load_or_build(items, cfg)
    queries = data.build_train_queries(cfg, items)
    if max_queries and max_queries < len(queries):
        queries = queries.sample(max_queries, random_state=cfg.seed).reset_index(drop=True)
    split = data.split_queries(queries, cfg)
    log.info(
        "Запросов: fit %d, early stop %d, hold-out %d", len(split.fit), len(split.early_stop), len(split.holdout)
    )

    # Популярность — только по fit: hold-out не должен подсказывать сам себе.
    popularity = data.item_popularity(split.fit, index.n_items)
    fit = pipeline.training_frame(split.fit, index, cfg, popularity, counted_in_popularity=True)
    es = pipeline.training_frame(split.early_stop, index, cfg, popularity, counted_in_popularity=False)
    cols = features.feature_columns(cfg)
    model = ranker.train(fit, es, cols, cfg)

    predicted, baselines, pools = pipeline.rank(
        split.holdout, index, cfg, model, popularity, collect_baselines=True
    )
    relevant = split.holdout["relevant"].tolist()

    report = {
        "config": cli.run_name(cfg),
        "queries": {"fit": len(split.fit), "early_stop": len(split.early_stop), "holdout": len(split.holdout)},
        "recall@50": {f"{r} top-50": recall_at_k(baselines[r], relevant) for r in cfg.retrievers},
        "pool_size_mean": float(np.mean([len(p) for p in pools.values()])) if pools else 0.0,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    report["recall@50"]["пул кандидатов (потолок)"] = recall_at_k(pools, relevant, k=10**9)
    report["recall@50"]["ранкер"] = recall_at_k(predicted, relevant)
    gain = model.booster_.feature_importance(importance_type="gain")
    report["feature_gain"] = dict(
        sorted(zip(cols, (gain / gain.sum()).round(4).tolist(), strict=True), key=lambda x: -x[1])
    )
    return report


def to_markdown(report: dict) -> str:
    lines = [
        f"# Валидация: {report['config']}",
        "",
        f"Запросов: fit {report['queries']['fit']}, early stop {report['queries']['early_stop']}, "
        f"hold-out {report['queries']['holdout']}. Средний размер пула: {report['pool_size_mean']:.0f}.",
        "",
        "| Метод | Recall@50 |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.4f} |" for name, value in report["recall@50"].items()]
    lines += ["", "Вклад признаков (доля gain):", "", "| Признак | Gain |", "|---|---|"]
    lines += [f"| {name} | {value:.3f} |" for name, value in report["feature_gain"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = cli.parser(__doc__.splitlines()[0]).parse_args()
    cfg = cli.config_from_args(args)
    report = run(cfg, args.max_queries)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    name = f"validation_{report['config']}"
    (out / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{name}.md").write_text(to_markdown(report), encoding="utf-8")
    print(pd.Series(report["recall@50"]).to_string())


if __name__ == "__main__":
    main()
