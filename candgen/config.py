from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("dataset")
    artifacts_dir: Path = Path("artifacts")

    # Стемминг. Отключается только в тестах, где нет nltk.
    stem: bool = True

    # Сколько кандидатов берёт каждый источник.
    k_full: int = 1000
    k_title: int = 1000
    k_char3: int = 1000
    k_full_loc: int = 300  # BM25 только по объявлениям из города запроса
    k_dense: int = 300

    use_dense: bool = True
    use_location: bool = True
    use_popularity: bool = True
    # Добавлять search_infm_params_text в текст запроса для BM25.
    # По умолчанию нет: эти параметры есть почти у всех объявлений категории
    # и размывают поиск. Они используются отдельным признаком.
    infm_in_query: bool = False

    dense_model: str = "intfloat/multilingual-e5-small"
    batch_size: int = 128  # запросов в одном матричном умножении BM25
    chunk_queries: int = 512  # запросов, для которых признаки держатся в памяти одновременно

    # Обучающие пары: все позитивы из пула + сложные и случайные негативы.
    n_hard_negatives: int = 100
    n_random_negatives: int = 100

    holdout_share: float = 0.2
    early_stop_share: float = 0.1
    seed: int = 42

    lgb_params: dict = field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [50],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_estimators": 2000,
            "verbose": -1,
        }
    )

    @property
    def retrievers(self) -> list[str]:
        names = ["full", "title", "char3"]
        if self.use_location:
            names.append("full_loc")
        if self.use_dense:
            names.append("dense")
        return names

    def k(self, retriever: str) -> int:
        return getattr(self, f"k_{retriever}")

    @property
    def missing_rank(self) -> int:
        """Ранг для кандидата, которого этот источник не нашёл."""
        return max(self.k(r) for r in self.retrievers)
