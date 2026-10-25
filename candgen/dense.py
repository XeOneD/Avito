"""Плотный поиск на multilingual-e5-small.

E5 обучалась с префиксами "query: " и "passage: " - без них качество падает.
Модель можно указать локальной папкой (DENSE_MODEL в Config), если huggingface.co недоступен.
"""

from functools import lru_cache

import numpy as np

from candgen.config import Config


@lru_cache(maxsize=1)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def _encode(texts: list[str], prefix: str, cfg: Config) -> np.ndarray:
    model = _model(cfg.dense_model)
    emb = model.encode(
        [prefix + t.strip() for t in texts],
        batch_size=cfg.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10_000,
    )
    return emb.astype(np.float32)


def encode_passages(texts: list[str], cfg: Config) -> np.ndarray:
    return _encode(texts, "passage: ", cfg)


def encode_queries(texts: list[str], cfg: Config) -> np.ndarray:
    return _encode(texts, "query: ", cfg)
