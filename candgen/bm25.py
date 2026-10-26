"""BM25 на разреженных матрицах scipy.

Весь индекс - одна матрица весов документ × терм: w(d, t) = idf(t) · tf·(k1+1) / (tf + k1·(1−b+b·|d|/avgdl)).
Скор запроса - сумма весов его термов, то есть произведение матриц. Это даёт два свойства:":
  - точный скор BM25 можно получить для любой пары запрос-объявление, а не только для top-K;
  - тот же расчёт даёт топ внутри подмножества документов (например, одного города).
idf как в Lucene: log(1 + (N − df + 0.5) / (df + 0.5)).
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, analyzer, k1: float = 1.5, b: float = 0.75):
        self.analyzer = analyzer
        self.k1 = k1
        self.b = b

    def fit(self, texts) -> "BM25":
        self.vectorizer = CountVectorizer(analyzer=self.analyzer, dtype=np.float32)
        tf = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean() if n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(avg_len, 1e-9))
        row_norm = np.repeat(norm, np.diff(tf.indptr)).astype(np.float32)

        weights = tf.copy()
        weights.data = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + row_norm)
        self.doc_term = weights  # документ × терм, для скоров отдельных пар
        self.term_doc = weights.T.tocsr()  # терм × документ, для скоров по всему корпусу
        return self

    @property
    def n_docs(self) -> int:
        return self.doc_term.shape[0]

    def encode(self, texts) -> sp.csr_matrix:
        """Запросы -> бинарная матрица термов. Повтор слова в запросе вес не удваивает."""
        q = self.vectorizer.transform(texts).tocsr()
        q.data[:] = 1.0
        return q

    def score_all(self, query_matrix: sp.csr_matrix) -> np.ndarray:
        """Скоры каждого запроса против каждого документа: (n_queries, n_docs)."""
        return (query_matrix @ self.term_doc).toarray()

    def score_pairs(self, query_matrix: sp.csr_matrix, q_idx: np.ndarray, d_idx: np.ndarray) -> np.ndarray:
        return rowwise_dot(query_matrix, self.doc_term, q_idx, d_idx)


def rowwise_dot(a: sp.csr_matrix, b: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, chunk: int = 200_000):
    """Скалярные произведения строк a[ia[k]] и b[ib[k]] без плотных матриц."""
    out = np.empty(len(ia), dtype=np.float32)
    for start in range(0, len(ia), chunk):
        stop = start + chunk
        prod = a[ia[start:stop]].multiply(b[ib[start:stop]])
        out[start:stop] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы k лучших документов с положительным скором, по убыванию.
    Ничьи разрешаются по индексу документа, чтобы результат был детерминированным."""
    if k >= len(scores):
        candidates = np.arange(len(scores))
    else:
        candidates = np.argpartition(-scores, k)[:k]
    candidates = candidates[scores[candidates] > 0]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order]
