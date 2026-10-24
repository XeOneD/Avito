"""Общие аргументы командной строки для validate и submit."""

import argparse
import logging
from pathlib import Path

from candgen.config import Config


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", type=Path, default=Path("dataset"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--no-dense", action="store_true", help="без e5 (быстрее, нужен только CPU)")
    p.add_argument("--no-location", action="store_true", help="без кандидатов из города запроса")
    p.add_argument("--no-popularity", action="store_true", help="без признака популярности объявления")
    p.add_argument("--infm-in-query", action="store_true", help="добавлять параметры поиска в текст запроса")
    p.add_argument("--dense-model", default="intfloat/multilingual-e5-small", help="имя на HF или локальная папка")
    p.add_argument("--max-queries", type=int, default=None, help="взять случайное подмножество train-запросов")
    return p


def config_from_args(args) -> Config:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    return Config(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        use_dense=not args.no_dense,
        use_location=not args.no_location,
        use_popularity=not args.no_popularity,
        infm_in_query=args.infm_in_query,
        dense_model=args.dense_model,
    )


def run_name(cfg: Config) -> str:
    parts = []
    if not cfg.use_dense:
        parts.append("no-dense")
    if not cfg.use_location:
        parts.append("no-location")
    if not cfg.use_popularity:
        parts.append("no-popularity")
    if cfg.infm_in_query:
        parts.append("infm-in-query")
    return "_".join(parts) or "full"
