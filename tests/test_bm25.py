import math

import numpy as np
import pytest

from candgen.bm25 import BM25, top_k
from candgen.text import CharNgramAnalyzer, WordAnalyzer

DOCS = ["ремонт холодильников", "ремонт ремонт стиральных машин", "вывоз мусора газель", ""]


def brute_force(query_terms, docs, k1=1.5, b=0.75):
    tokenized = [d.split() for d in docs]
    n = len(docs)
    avgdl = sum(map(len, tokenized)) / n
    scores = []
    for doc in tokenized:
        s = 0.0
        for term in set(query_terms):
            df = sum(term in d for d in tokenized)
            tf = doc.count(term)
            if tf == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(s)
    return np.array(scores)


@pytest.fixture
def bm25():
    return BM25(WordAnalyzer(stem=False)).fit(DOCS)


def test_scores_match_formula(bm25):
    scores = bm25.score_all(bm25.encode(["ремонт машин ремонт"]))[0]
    np.testing.assert_allclose(scores, brute_force(["ремонт", "машин"], DOCS), rtol=1e-5)


def test_score_pairs_equals_full_scoring(bm25):
    q = bm25.encode(["ремонт холодильников", "газель"])
    full = bm25.score_all(q)
    qi, di = np.array([0, 0, 1, 1]), np.array([0, 2, 2, 1])
    np.testing.assert_allclose(bm25.score_pairs(q, qi, di), full[qi, di], rtol=1e-6)


def test_unknown_words_give_zero(bm25):
    assert bm25.score_all(bm25.encode(["абракадабра"])).sum() == 0


def test_top_k_skips_zero_scores_and_breaks_ties_by_index():
    scores = np.array([0.0, 2.0, 5.0, 2.0, 0.0])
    assert top_k(scores, 10).tolist() == [2, 1, 3]
    assert top_k(scores, 2).tolist() == [2, 1]


def test_char_ngrams_ignore_spaces_and_case():
    analyzer = CharNgramAnalyzer(3)
    assert analyzer("1 С") == analyzer("1с") == ["1с"]
    assert analyzer("Ёлка") == ["елк", "лка"]


def test_word_analyzer_without_stemming():
    assert WordAnalyzer(stem=False)("Ремонт, ёлки и 3D-печать!") == ["ремонт", "елки", "3d", "печать"]
