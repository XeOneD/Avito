import numpy as np


def recall_at_k(predicted: dict[int, np.ndarray], relevant: list[np.ndarray], k: int = 50) -> float:
    """Средний по запросам Recall@k. Запрос без кандидатов даёт 0, а не выпадает из среднего."""
    values = []
    for q, rel in enumerate(relevant):
        pred = predicted.get(q, np.array([], dtype=int))[:k]
        values.append(len(np.intersect1d(pred, rel)) / len(rel))
    return float(np.mean(values)) if values else float("nan")
