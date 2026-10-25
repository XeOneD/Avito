"""Маленький синтетический датасет той же схемы, что у Авито: для тестов и сквозного прогона
без настоящих данных. Объявления разбиты по темам и городам, запросы выбирают объявления
своей темы, чаще - в своём городе."""

import numpy as np
import pandas as pd

TOPICS = {
    "ремонт холодильников": ["ремонт", "холодильник", "мастер", "выезд"],
    "вывоз мусора": ["вывоз", "мусор", "контейнер", "газель"],
    "автоподбор": ["автоподбор", "проверка", "автомобиль", "диагностика"],
    "маникюр": ["маникюр", "гель", "покрытие", "ногти"],
    "репетитор математика": ["репетитор", "математика", "огэ", "егэ"],
    "натяжные потолки": ["потолки", "натяжные", "монтаж", "замер"],
    "грузоперевозки": ["перевозка", "грузчики", "переезд", "газель"],
    "фотограф свадьба": ["фотограф", "свадьба", "фотосессия", "съемка"],
}
LOCATIONS = [637640, 640860, 652430]
CATEGORY = 114


def make(n_items_per_topic: int = 40, n_queries: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    rows = []
    for t, topic in enumerate(topics):
        words = TOPICS[topic]
        for i in range(n_items_per_topic):
            title_words = rng.choice(words, size=2, replace=False)
            rows.append(
                {
                    "item_id": f"{t:02x}{i:014x}",
                    "item_title_raw": " ".join(title_words).capitalize(),
                    "item_description_raw": " ".join(rng.choice(words, size=6))
                    + f" опыт {rng.integers(1, 20)} лет",
                    "item_infm_params_text": f"Вид услуги {topic.split()[0].capitalize()}",
                    "item_category_id": CATEGORY,
                    "item_microcat_id": 1000 + t,
                    "item_location_id": int(rng.choice(LOCATIONS)),
                    "item_price": f"{rng.integers(300, 5000)}.000000000000000",
                    "item_rating": float(rng.uniform(3, 5)),
                    "item_rating_reviews_count": float(rng.integers(0, 200)),
                    "_topic": t,
                }
            )
    items = pd.DataFrame(rows)

    train_rows, bench_rows = [], []
    for n in range(n_queries):
        t = int(rng.integers(len(topics)))
        loc = int(rng.choice(LOCATIONS))
        text = topics[t] if rng.random() < 0.5 else " ".join(rng.choice(TOPICS[topics[t]], size=2, replace=False))
        same_topic = items[items["_topic"] == t]
        local = same_topic[same_topic["item_location_id"] == loc]
        pool = local if len(local) and rng.random() < 0.8 else same_topic
        for item in pool.sample(int(rng.integers(1, 4)), random_state=int(rng.integers(1 << 31))).itertuples():
            train_rows.append(
                {
                    "search_query": text,
                    "search_location_id": loc,
                    "search_is_delivery_search": 0,
                    "search_infm_params_text": "",
                    "search_category": CATEGORY,
                    "item_id": item.item_id,
                }
            )
        if n < 60:
            bench_rows.append(
                {
                    "query_id": f"Q{n:015d}",
                    "search_query": text,
                    "search_location_id": loc,
                    "search_is_delivery_search": 0,
                    "search_infm_params_text": "",
                    "search_category": CATEGORY,
                }
            )
    return items.drop(columns="_topic"), pd.DataFrame(train_rows), pd.DataFrame(bench_rows)


def write(path, **kwargs) -> None:
    items, train, bench = make(**kwargs)
    path.mkdir(parents=True, exist_ok=True)
    items.to_parquet(path / "benchmark_items.parquet")
    train.to_parquet(path / "train.parquet")
    bench.to_parquet(path / "benchmark_queries.parquet")
