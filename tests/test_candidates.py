import numpy as np

from candgen import candidates, features


def test_ranks_are_consistent_with_scores(train_queries, index, cfg):
    queries = train_queries.head(20)
    cand = candidates.generate(queries, index, cfg)
    missing = cfg.missing_rank
    for _, group in cand.groupby("q"):
        found = group[group["rank_full"] < missing].sort_values("rank_full")
        # Ранги идут подряд с нуля, и чем меньше ранг, тем выше скор.
        assert found["rank_full"].tolist() == list(range(len(found)))
        assert (np.diff(found["bm25_full"].to_numpy()) <= 1e-6).all()


def test_scores_are_exact_for_every_candidate(train_queries, index, cfg):
    queries = train_queries.head(20)
    cand = candidates.generate(queries, index, cfg)
    q = index.full.encode(queries["search_query"].tolist())
    exact = index.full.score_pairs(q, cand["q"].to_numpy(), cand["item"].to_numpy())
    np.testing.assert_allclose(cand["bm25_full"].to_numpy(), exact, rtol=1e-5)


def test_location_source_returns_only_same_city(train_queries, index, cfg):
    queries = train_queries.head(30)
    cand = candidates.generate(queries, index, cfg)
    local = cand[cand["rank_full_loc"] < cfg.missing_rank]
    item_loc = index.items["item_location_id"].to_numpy()[local["item"]]
    query_loc = queries["search_location_id"].to_numpy()[local["q"]]
    assert len(local) and (item_loc == query_loc).all()


def test_training_sample_has_no_positives_outside_pool(train_queries, index, cfg):
    queries = train_queries.head(40)
    full = candidates.generate(queries, index, cfg)
    sampled = candidates.generate(queries, index, cfg, sample_for_training=True)
    pool = set(zip(full["q"], full["item"], strict=True))
    assert set(zip(sampled["q"], sampled["item"], strict=True)) <= pool

    labeled = features.add_labels(sampled, queries)
    per_query = labeled.groupby("q")["label"].agg(["sum", "size"])
    assert (per_query["size"] - per_query["sum"] <= cfg.n_hard_negatives + cfg.n_random_negatives).all()


def test_offset_shifts_query_ids(train_queries, index, cfg):
    queries = train_queries.head(5)
    cand = candidates.generate(queries, index, cfg, q_offset=100)
    assert cand["q"].min() >= 100 and cand["q"].max() < 105
