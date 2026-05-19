"""Tests for the eval runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from filings_analyst.eval.runner import (
    EvalRunner,
    GoldenItem,
    ItemResult,
    MetricResult,
)


GOLDEN_SAMPLE = [
    {
        "id": "aapl-001",
        "ticker": "AAPL",
        "accession_no": "0000320193-25-000079",
        "question": "What does Apple say about supply chain risk?",
        "type": "easy",
        "expected_section": "Risk Factors",
        "reference_answer": "Apple identifies concentration risk in Asia.",
        "scope": "in-scope",
        "notes": "ok",
    },
    {
        "id": "aapl-007",
        "ticker": "AAPL",
        "accession_no": "0000320193-25-000079",
        "question": "What did Tim Cook announce at WWDC?",
        "type": "out-of-scope",
        "expected_section": None,
        "reference_answer": "Out of scope.",
        "scope": "out-of-scope",
        "notes": "ok",
    },
    {
        "id": "msft-001",
        "ticker": "MSFT",
        "accession_no": "0000950170-25-100235",
        "question": "What are Microsoft's segments?",
        "type": "easy",
        "expected_section": "Business",
        "reference_answer": "Three segments.",
        "scope": "in-scope",
        "notes": "ok",
    },
]


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    import yaml

    p = tmp_path / "golden.yaml"
    p.write_text(yaml.safe_dump(GOLDEN_SAMPLE), encoding="utf-8")
    return p


def _fake_rag(answer="Apple says supply chain is concentrated. [Risk Factors §0]"):
    rag = MagicMock()
    rag.ask_filing.return_value = {
        "question": "Q?",
        "answer": answer,
        "cited_chunks": [
            {"section": "Risk Factors", "chunk_idx": 0, "text": "Supply chain concentration."}
        ],
        "provider": "anthropic_api",
    }
    return rag


def _fake_grader(responses=None):
    """LLMProvider mock returning the queued responses cyclically."""
    grader = MagicMock()
    grader.available = True
    grader.provider = "anthropic_api"
    if responses is None:
        # Default: faithfulness JSON, then relevancy infer + compare, then
        # precision yes, then recall yes — covers a full grading cycle.
        responses = [
            json.dumps({"claims": ["c1"], "supported": [True]}),  # faithfulness
            "What about supply chain?",                            # relevancy infer
            json.dumps({"score": 1.0, "reason": "same"}),          # relevancy compare
            json.dumps({"relevant": True}),                        # precision (1 chunk)
            json.dumps({"supported": True}),                       # recall (1 sentence)
        ]
    grader.generate = MagicMock(side_effect=lambda *a, **k: next(_grader_iter))
    _grader_iter = iter(_cycle(responses))
    grader.generate.side_effect = _grader_iter
    return grader


def _cycle(items):
    while True:
        for it in items:
            yield it


def test_load_golden_set_from_path(tmp_path: Path, yaml_path: Path):
    runner = EvalRunner(cache_dir=tmp_path)
    items = runner.load_golden_set(yaml_path)
    assert len(items) == 3
    assert items[0].id == "aapl-001"
    assert items[1].scope == "out-of-scope"


def test_load_golden_set_default_path(tmp_path: Path):
    """Loading without a path reads the packaged golden_set.yaml."""
    runner = EvalRunner(cache_dir=tmp_path)
    items = runner.load_golden_set()
    assert len(items) >= 30  # we shipped 35
    ids = {i.id for i in items}
    assert "aapl-001" in ids
    # All entries declare a scope.
    assert all(i.scope in ("in-scope", "out-of-scope") for i in items)


def test_run_applies_sample_and_only_types(tmp_path: Path, yaml_path: Path):
    runner = EvalRunner(
        rag=_fake_rag(), grader_llm=_fake_grader(), cache_dir=tmp_path
    )
    items = runner.load_golden_set(yaml_path)
    result = runner.run(items, sample=1, progress=False)
    assert result.n_run == 1
    assert result.n_total == 3

    result2 = runner.run(items, only_types={"out-of-scope"}, progress=False)
    assert result2.n_run == 1
    assert result2.per_item_results[0].scope == "out-of-scope"


def test_run_caches_results(tmp_path: Path, yaml_path: Path):
    grader = _fake_grader()
    runner = EvalRunner(rag=_fake_rag(), grader_llm=grader, cache_dir=tmp_path)
    items = runner.load_golden_set(yaml_path)

    runner.run([items[0]], progress=False)
    first_call_count = grader.generate.call_count
    assert first_call_count > 0

    # Re-run with cache enabled — grader should not be called again.
    runner.run([items[0]], progress=False)
    assert grader.generate.call_count == first_call_count


def test_run_no_cache_re_grades(tmp_path: Path, yaml_path: Path):
    grader = _fake_grader()
    runner = EvalRunner(rag=_fake_rag(), grader_llm=grader, cache_dir=tmp_path)
    items = runner.load_golden_set(yaml_path)
    runner.run([items[0]], progress=False)
    n_after_first = grader.generate.call_count
    runner.run([items[0]], skip_cached=False, progress=False)
    assert grader.generate.call_count > n_after_first


def test_run_aggregate_metrics_are_means(tmp_path: Path, yaml_path: Path):
    runner = EvalRunner(rag=_fake_rag(), grader_llm=_fake_grader(), cache_dir=tmp_path)
    items = runner.load_golden_set(yaml_path)
    result = runner.run(items, progress=False)
    # All five metrics show up.
    assert set(result.aggregate_metrics.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "refusal_correctness",
    }
    for name, val in result.aggregate_metrics.items():
        assert 0.0 <= val <= 1.0


def test_run_handles_rag_error_gracefully(tmp_path: Path, yaml_path: Path):
    bad_rag = MagicMock()
    bad_rag.ask_filing.side_effect = RuntimeError("vector store gone")
    runner = EvalRunner(rag=bad_rag, grader_llm=_fake_grader(), cache_dir=tmp_path)
    items = runner.load_golden_set(yaml_path)
    result = runner.run([items[0]], progress=False)
    assert result.per_item_results[0].error is not None
    # Per-item exception turns every metric into 0.0 with a reason.
    for m in result.per_item_results[0].metrics.values():
        assert m.score == 0.0


def test_item_result_round_trip(tmp_path: Path):
    ir = ItemResult(
        item_id="x",
        ticker="AAPL",
        type="easy",
        scope="in-scope",
        question="Q?",
        answer="A",
        cited_chunks=[{"section": "S", "chunk_idx": 0, "text": "t"}],
        metrics={"faithfulness": MetricResult(score=0.5, reason="r", raw_grade=None)},
    )
    d = ir.to_dict()
    ir2 = ItemResult.from_dict(d)
    assert ir2.metrics["faithfulness"].score == 0.5
    assert ir2.item_id == "x"


def test_per_type_and_per_ticker_breakdowns(tmp_path: Path, yaml_path: Path):
    runner = EvalRunner(rag=_fake_rag(), grader_llm=_fake_grader(), cache_dir=tmp_path)
    items = runner.load_golden_set(yaml_path)
    result = runner.run(items, progress=False)

    types_b = result.per_type_breakdown()
    assert set(types_b.keys()) == {"easy", "out-of-scope"}
    assert types_b["easy"]["n"] == 2
    assert "per_metric" in types_b["easy"]

    tickers_b = result.per_ticker_breakdown()
    assert set(tickers_b.keys()) == {"AAPL", "MSFT"}
    assert tickers_b["AAPL"]["n"] == 2
