import pytest

from candgen import data, indexes
from candgen.config import Config
from tests import synthetic


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("avito")
    synthetic.write(root / "dataset")
    return Config(
        data_dir=root / "dataset",
        artifacts_dir=root / "artifacts",
        stem=False,
        use_dense=False,
        k_full=60,
        k_title=60,
        k_char3=60,
        k_full_loc=30,
        n_hard_negatives=20,
        n_random_negatives=20,
        batch_size=16,
        chunk_queries=64,
        lgb_params={**Config().lgb_params, "n_estimators": 50, "min_child_samples": 5},
    )


@pytest.fixture(scope="session")
def items(cfg):
    return data.load_items(cfg)


@pytest.fixture(scope="session")
def index(items, cfg):
    return indexes.build(items, cfg)


@pytest.fixture(scope="session")
def train_queries(cfg, items):
    return data.build_train_queries(cfg, items)
