"""Markdown report generator for eval results."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from .runner import EvalRunResult, ItemResult


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def format_summary_table(result: EvalRunResult) -> str:
    lines = ["| Metric | Score |", "|--------|-------|"]
    for name, score in result.aggregate_metrics.items():
        lines.append(f"| {name} | {score:.3f} |")
    return "\n".join(lines)


def format_breakdowns(result: EvalRunResult) -> str:
    parts: list[str] = []

    parts.append("### Per-type breakdown\n")
    parts.append("| Type | n | Mean (all metrics) | faithfulness | answer_relevancy | context_precision | context_recall | refusal_correctness |")
    parts.append("|------|---|--------------------|--------------|------------------|-------------------|----------------|---------------------|")
    types_b = result.per_type_breakdown()
    for tname in sorted(types_b.keys()):
        b = types_b[tname]
        pm = b["per_metric"]
        parts.append(
            f"| {tname} | {b['n']} | {b['mean_score']:.3f} | "
            f"{pm.get('faithfulness', 0.0):.3f} | "
            f"{pm.get('answer_relevancy', 0.0):.3f} | "
            f"{pm.get('context_precision', 0.0):.3f} | "
            f"{pm.get('context_recall', 0.0):.3f} | "
            f"{pm.get('refusal_correctness', 0.0):.3f} |"
        )

    parts.append("\n### Per-ticker breakdown\n")
    parts.append("| Ticker | n | Mean (all metrics) | faithfulness | answer_relevancy | context_precision | context_recall | refusal_correctness |")
    parts.append("|--------|---|--------------------|--------------|------------------|-------------------|----------------|---------------------|")
    tickers_b = result.per_ticker_breakdown()
    for t in sorted(tickers_b.keys()):
        b = tickers_b[t]
        pm = b["per_metric"]
        parts.append(
            f"| {t} | {b['n']} | {b['mean_score']:.3f} | "
            f"{pm.get('faithfulness', 0.0):.3f} | "
            f"{pm.get('answer_relevancy', 0.0):.3f} | "
            f"{pm.get('context_precision', 0.0):.3f} | "
            f"{pm.get('context_recall', 0.0):.3f} | "
            f"{pm.get('refusal_correctness', 0.0):.3f} |"
        )
    return "\n".join(parts)


def format_worst_items(result: EvalRunResult, k: int = 3) -> str:
    """For each metric, list the k items with the worst scores.

    Ties are broken by item_id (stable) so the report is deterministic.
    """
    if not result.per_item_results:
        return "_No items to rank._"
    parts: list[str] = []
    metric_names = sorted(result.aggregate_metrics.keys())
    for metric_name in metric_names:
        scored: list[tuple[float, ItemResult]] = []
        for it in result.per_item_results:
            m = it.metrics.get(metric_name)
            if m is None:
                continue
            scored.append((m.score, it))
        scored.sort(key=lambda x: (x[0], x[1].item_id))
        worst = scored[:k]
        parts.append(f"\n#### {metric_name}\n")
        if not worst:
            parts.append("_No graded items._")
            continue
        for score, it in worst:
            reason = it.metrics[metric_name].reason
            ans_preview = (it.answer or "(no answer)").replace("\n", " ")
            if len(ans_preview) > 160:
                ans_preview = ans_preview[:160] + "..."
            parts.append(
                f"- `{it.item_id}` (score={score:.2f}, type={it.type}, ticker={it.ticker}) — "
                f"{reason}\n  - Q: {it.question}\n  - A: {ans_preview}"
            )
    return "\n".join(parts)


def write_full_report(result: EvalRunResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# filings-analyst eval report",
        "",
        f"- Run started: `{result.started_at}`",
        f"- Duration: `{result.duration_s:.1f}s`",
        f"- Provider: `{result.provider}`",
        f"- Items run: `{result.n_run}` / `{result.n_total}` total",
        f"- Git SHA: `{_git_sha()}`",
        "",
        "## Aggregate metrics",
        "",
        format_summary_table(result),
        "",
        "## Breakdowns",
        "",
        format_breakdowns(result),
        "",
        "## Worst-3 items per metric",
        "",
        format_worst_items(result, k=3),
        "",
        "## Methodology notes",
        "",
        "- Golden set is hand-curated, 35 items across AAPL / MSFT / JPM / BAC / XOM.",
        "- `refusal_correctness` is deterministic (substring + citation regex). The other four metrics route grader prompts through `LLMProvider`.",
        "- For out-of-scope items the system *should* refuse; faithfulness, "
        "  answer_relevancy, context_precision, and context_recall short-circuit "
        "  to 1.0 when the refusal is detected, on the principle that a correct "
        "  refusal is not a retrieval failure.",
        "- Re-run with: `filings-analyst eval`.",
        "",
    ]
    path.write_text("\n".join(header), encoding="utf-8")


__all__ = (
    "format_summary_table",
    "format_breakdowns",
    "format_worst_items",
    "write_full_report",
)
