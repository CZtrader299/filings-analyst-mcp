"""Tests for the markdown report generator."""

from __future__ import annotations

from pathlib import Path

from filings_analyst.eval.report import (
    format_breakdowns,
    format_summary_table,
    format_worst_items,
    write_full_report,
)
from filings_analyst.eval.runner import EvalRunResult, ItemResult, MetricResult


def _mk_item(item_id, ticker, type_, scope, scores):
    metrics = {
        name: MetricResult(score=score, reason=f"{name} reason for {item_id}")
        for name, score in scores.items()
    }
    return ItemResult(
        item_id=item_id,
        ticker=ticker,
        type=type_,
        scope=scope,
        question=f"question for {item_id}?",
        answer=f"answer for {item_id}",
        cited_chunks=[],
        metrics=metrics,
    )


def _mk_result(items):
    agg: dict[str, list[float]] = {}
    for it in items:
        for name, m in it.metrics.items():
            agg.setdefault(name, []).append(m.score)
    aggregate = {name: sum(v) / len(v) for name, v in agg.items()}
    return EvalRunResult(
        per_item_results=items,
        aggregate_metrics=aggregate,
        provider="anthropic_api",
        started_at="2026-05-19T12:00:00+00:00",
        duration_s=42.0,
        n_total=len(items),
        n_run=len(items),
    )


def test_summary_table_renders_all_metrics():
    items = [
        _mk_item(
            "a-1",
            "AAPL",
            "easy",
            "in-scope",
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.9,
                "context_precision": 0.7,
                "context_recall": 0.6,
                "refusal_correctness": 1.0,
            },
        )
    ]
    out = format_summary_table(_mk_result(items))
    assert "| Metric | Score |" in out
    assert "faithfulness" in out
    assert "0.800" in out


def test_breakdowns_include_types_and_tickers():
    items = [
        _mk_item("a-1", "AAPL", "easy", "in-scope", {"faithfulness": 1.0}),
        _mk_item("a-2", "AAPL", "hard", "in-scope", {"faithfulness": 0.5}),
        _mk_item("m-1", "MSFT", "easy", "in-scope", {"faithfulness": 0.0}),
    ]
    out = format_breakdowns(_mk_result(items))
    assert "Per-type breakdown" in out
    assert "Per-ticker breakdown" in out
    assert "easy" in out
    assert "hard" in out
    assert "AAPL" in out
    assert "MSFT" in out


def test_worst_items_sorts_ascending_by_score():
    items = [
        _mk_item("a-1", "AAPL", "easy", "in-scope", {"faithfulness": 1.0}),
        _mk_item("a-2", "AAPL", "easy", "in-scope", {"faithfulness": 0.2}),
        _mk_item("a-3", "AAPL", "easy", "in-scope", {"faithfulness": 0.5}),
        _mk_item("a-4", "AAPL", "easy", "in-scope", {"faithfulness": 0.0}),
    ]
    out = format_worst_items(_mk_result(items), k=2)
    # First listed in the "faithfulness" block must be lowest-scoring (a-4 at 0.0).
    block_pos = out.find("faithfulness")
    assert block_pos >= 0
    after = out[block_pos:]
    assert after.find("a-4") < after.find("a-2")  # a-4 (0.0) before a-2 (0.2)


def test_worst_items_handles_empty_result():
    empty = EvalRunResult(
        per_item_results=[],
        aggregate_metrics={},
        provider="none",
        started_at="now",
        duration_s=0.0,
        n_total=0,
        n_run=0,
    )
    out = format_worst_items(empty)
    assert "No items" in out


def test_write_full_report_creates_parseable_markdown(tmp_path: Path):
    items = [
        _mk_item(
            "a-1",
            "AAPL",
            "easy",
            "in-scope",
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.9,
                "context_precision": 0.7,
                "context_recall": 0.6,
                "refusal_correctness": 1.0,
            },
        ),
        _mk_item(
            "a-2",
            "AAPL",
            "out-of-scope",
            "out-of-scope",
            {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
                "refusal_correctness": 1.0,
            },
        ),
    ]
    out_path = tmp_path / "report.md"
    write_full_report(_mk_result(items), out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "# filings-analyst eval report" in text
    assert "Aggregate metrics" in text
    assert "Per-type breakdown" in text
    assert "Per-ticker breakdown" in text
    assert "Worst-3 items per metric" in text
    assert "Methodology notes" in text
