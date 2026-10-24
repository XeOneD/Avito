"""Сквозной прогон validate и submit на синтетике. Проверяет, что всё связано
правильно, а не качество: на игрушечных данных цифры ничего не значат."""

import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from candgen import submit, validate  # noqa: E402


def test_validate_runs_and_ranker_beats_bm25(cfg):
    report = validate.run(cfg)
    recalls = report["recall@50"]
    assert all(0 <= v <= 1 for v in recalls.values())
    assert recalls["ранкер"] <= recalls["пул кандидатов (потолок)"] + 1e-9
    assert recalls["ранкер"] >= recalls["full top-50"] - 0.05
    assert validate.to_markdown(report).startswith("# Валидация")


def test_submit_writes_valid_answer(cfg, tmp_path, items):
    out = tmp_path / "answer.csv"
    submit.run(cfg, out=out)
    answer = pd.read_csv(out, dtype=str)
    assert len(answer) == 60
    assert answer["answer"].str.split(" ").map(len).eq(50).all()
