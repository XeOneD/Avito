import numpy as np

from candgen import data


def test_relevant_items_are_unique_and_in_corpus(train_queries, items):
    for rel in train_queries["relevant"]:
        assert len(rel) == len(np.unique(rel))
        assert rel.min() >= 0 and rel.max() < len(items)


def test_split_has_no_shared_query_texts(train_queries, cfg):
    split = data.split_queries(train_queries, cfg)
    parts = [set(p["search_query"].str.strip().str.lower()) for p in (split.fit, split.early_stop, split.holdout)]
    assert not parts[0] & parts[1] and not parts[0] & parts[2] and not parts[1] & parts[2]
    assert len(split.fit) + len(split.early_stop) + len(split.holdout) == len(train_queries)


def test_popularity_counts_queries(train_queries, items):
    pop = data.item_popularity(train_queries, len(items))
    assert pop.sum() == sum(len(r) for r in train_queries["relevant"])


def test_query_text_is_the_same_function_everywhere(train_queries, cfg):
    assert data.query_text(train_queries, cfg) == train_queries["search_query"].tolist()
