"""Run a stratified 10-item sample of the eval golden set.

Why a sample rather than the full 35-item set?
---------------------------------------------
Each item invokes the RAG pipeline once plus four LLM-graded metrics
(faithfulness, answer_relevancy, context_precision, context_recall) via
`claude -p`. End-to-end that's ~115s per item, ~67 min for the full
35-item run — too long for a single session.

This script runs a deterministic, stratified sample chosen to spread
across all five tickers and all three question types (easy / hard /
out-of-scope). The numbers from this run are the ones quoted in the
project README's Evaluation section — no hand-picked or fabricated
scores.

To reproduce the full 35-item run, use the CLI:

    filings-analyst eval

To reproduce just this sample:

    python scripts/run_sample_eval.py

The per-item grades land in
``~/.filings_analyst_cache/eval_cache/<item_id>_<provider>.json`` so
later re-runs are cache hits and finish in seconds.
"""

from __future__ import annotations

from pathlib import Path

from filings_analyst.eval.runner import EvalRunner
from filings_analyst.eval.report import write_full_report


# Stratified sample: 4 easy / 3 hard / 3 out-of-scope across all five
# tickers. `aapl-001` is intentionally included because it's already
# cached from the previous infrastructure-build session and re-using it
# costs nothing.
SAMPLE_IDS: list[str] = [
    "aapl-001",  # easy, AAPL — supply-chain risks
    "aapl-005",  # hard, AAPL — antitrust + Services synthesis
    "aapl-006",  # out-of-scope, AAPL — exec comp lives in proxy
    "msft-001",  # easy, MSFT — operating segments
    "msft-004",  # hard, MSFT — AI framing Business vs Risk Factors
    "jpm-001",   # easy, JPM — business segments
    "jpm-005",   # hard, JPM — operational + AI + third-party risk
    "jpm-007",   # out-of-scope, JPM — current P/B ratio (market data)
    "bac-001",   # easy, BAC — operating segments
    "xom-006",   # out-of-scope, XOM — real-time WTI spot price
]


def main() -> None:
    runner = EvalRunner()
    full = runner.load_golden_set()
    by_id = {it.id: it for it in full}

    missing = [i for i in SAMPLE_IDS if i not in by_id]
    if missing:
        raise SystemExit(f"Sample IDs not in golden set: {missing}")

    sample = [by_id[i] for i in SAMPLE_IDS]
    print(f"Running stratified sample of {len(sample)} items "
          f"out of {len(full)} total in the golden set.")
    for it in sample:
        print(f"  - {it.id:10s} {it.type:13s} {it.ticker}")
    print()

    result = runner.run(sample, skip_cached=True)

    print()
    print("=== Aggregate metrics ===")
    for name, score in result.aggregate_metrics.items():
        print(f"  {name:24s} {score:.3f}")
    print()
    print("=== Per-type breakdown ===")
    for tname, b in sorted(result.per_type_breakdown().items()):
        print(f"  {tname:14s} n={b['n']:<3d} mean={b['mean_score']:.3f}")
    print()
    print("=== Per-ticker breakdown ===")
    for t, b in sorted(result.per_ticker_breakdown().items()):
        print(f"  {t:6s} n={b['n']:<3d} mean={b['mean_score']:.3f}")
    print()

    out_path = Path(__file__).resolve().parents[1] / "eval_reports" / "2026-05-19_sample_eval.md"
    write_full_report(result, out_path)
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
