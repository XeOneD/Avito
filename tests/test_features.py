import numpy as np
import pandas as pd

from candgen import candidates, data, features


def _frame(queries, index, cfg, popularity, sample):
    cand = candidates.generate(queries, index, cfg, sample_for_training=sample)
    cand = features.add_labels(cand, queries)
    return features.add_features(cand, queries, index, cfg, popularity)


def test_same_pair_gets_same_features_in_training_and_inference(train_queries, index, cfg, items):
    """Главная причина старого разрыва val/платформа: признаки считались по-разному."""
    queries = train_queries.head(30)
    popularity = data.item_popularity(train_queries, len(items))
    train_frame = _frame(queries, index, cfg, popularity, sample=True)
    infer_frame = _frame(queries, index, cfg, popularity, sample=False)
    cols = features.feature_columns(cfg)
    merged = train_frame.merge(infer_frame, on=["q", "item"], suffixes=("_tr", "_inf"))
    assert len(merged) == len(train_frame)
    for col in cols:
        np.testing.assert_allclose(merged[f"{col}_tr"], merged[f"{col}_inf"], rtol=1e-6, err_msg=col)


def test_no_all_zero_rows(train_queries, index, cfg, items):
    frame = _frame(train_queries.head(30), index, cfg, None, sample=True)
    values = frame[features.feature_columns(cfg)].to_numpy()
    assert (np.abs(values).sum(axis=1) > 0).all()


def test_overlap_features_are_ratios(train_queries, index, cfg):
    frame = _frame(train_queries.head(30), index, cfg, None, sample=False)
    for col in ["overlap_recall", "overlap_jaccard", "title_recall", "desc_recall", "char3_jaccard"]:
        assert frame[col].between(0, 1 + 1e-6).all(), col


def test_popularity_leave_one_out(train_queries, index, cfg, items):
    queries = train_queries.head(30)
    popularity = data.item_popularity(queries, len(items))
    cand = features.add_labels(candidates.generate(queries, index, cfg), queries)
    cand["is_train_query"] = np.int8(1)
    frame = features.add_features(cand, queries, index, cfg, popularity)
    expected = np.log1p(popularity[frame["item"]] - frame["label"])
    np.testing.assert_allclose(frame["item_popularity"], expected, rtol=1e-6)


def test_cat_match_uses_item_category(train_queries, index, cfg):
    frame = _frame(train_queries.head(10), index, cfg, None, sample=False)
    assert frame["cat_match"].eq(1).all()  # в синтетике всё в категории 114


def test_labels_mark_relevant_items():
    queries = pd.DataFrame({"relevant": [np.array([1, 2]), np.array([5])]})
    cand = pd.DataFrame({"q": np.array([0, 0, 1, 1], np.int32), "item": np.array([1, 3, 5, 2], np.int32)})
    assert features.add_labels(cand, queries)["label"].tolist() == [1, 0, 1, 0]
