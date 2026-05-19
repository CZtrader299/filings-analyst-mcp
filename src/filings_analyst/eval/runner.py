"""Eval harness orchestrator.

Loads the YAML golden set, runs each item through the RAG pipeline,
grades the output with every registered metric, caches per-item
results to disk so re-runs skip already-graded items, and aggregates
into per-metric / per-type / per-ticker means.

Design choices worth flagging:

* The cache key is ``{item_id}_{provider_name}.json``. If we change
  provider (e.g., flip from claude_cli to anthropic_api) the cache
  rightly invalidates because the system under test changed.
* Cache contents include the full per-metric output so the report
  generator can compute "worst items per metric" without re-running.
* ``run()`` swallows per-item exceptions and records them as a 0.0
  score with the exception message in the reason — one broken item
  must not bring down a 35-item eval.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib.resources as _resources
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .. import config
from ..providers import LLMProvider
from ..rag import FilingRAG
from . import metrics as _metrics


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GoldenItem:
    id: str
    ticker: str
    accession_no: str
    question: str
    type: str  # easy | hard | out-of-scope
    expected_section: Optional[str]
    reference_answer: str
    scope: str  # in-scope | out-of-scope
    notes: str = ""


@dataclass
class MetricResult:
    score: float
    reason: str
    raw_grade: Optional[str] = None


@dataclass
class ItemResult:
    item_id: str
    ticker: str
    type: str
    scope: str
    question: str
    answer: Optional[str]
    cited_chunks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "ticker": self.ticker,
            "type": self.type,
            "scope": self.scope,
            "question": self.question,
            "answer": self.answer,
            "cited_chunks": self.cited_chunks,
            "metrics": {
                name: dataclasses.asdict(m) for name, m in self.metrics.items()
            },
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ItemResult":
        metrics_map = {
            name: MetricResult(**vals) for name, vals in (d.get("metrics") or {}).items()
        }
        return cls(
            item_id=d["item_id"],
            ticker=d["ticker"],
            type=d["type"],
            scope=d["scope"],
            question=d["question"],
            answer=d.get("answer"),
            cited_chunks=d.get("cited_chunks") or [],
            metrics=metrics_map,
            error=d.get("error"),
        )


@dataclass
class EvalRunResult:
    per_item_results: list[ItemResult]
    aggregate_metrics: dict[str, float]
    provider: str
    started_at: str
    duration_s: float
    n_total: int
    n_run: int

    def per_type_breakdown(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[ItemResult]] = {}
        for r in self.per_item_results:
            buckets.setdefault(r.type, []).append(r)
        out: dict[str, dict[str, Any]] = {}
        for k, items in buckets.items():
            out[k] = {
                "n": len(items),
                "mean_score": _mean_across_metrics(items),
                "per_metric": _mean_per_metric(items),
            }
        return out

    def per_ticker_breakdown(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[ItemResult]] = {}
        for r in self.per_item_results:
            buckets.setdefault(r.ticker, []).append(r)
        out: dict[str, dict[str, Any]] = {}
        for k, items in buckets.items():
            out[k] = {
                "n": len(items),
                "mean_score": _mean_across_metrics(items),
                "per_metric": _mean_per_metric(items),
            }
        return out


def _mean_across_metrics(items: list[ItemResult]) -> float:
    vals: list[float] = []
    for it in items:
        for m in it.metrics.values():
            vals.append(m.score)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _mean_per_metric(items: list[ItemResult]) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for it in items:
        for name, m in it.metrics.items():
            per.setdefault(name, []).append(m.score)
    return {
        name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in per.items()
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_DEFAULT_GOLDEN_FILENAME = "golden_set.yaml"


class EvalRunner:
    """Run the eval, with disk caching."""

    def __init__(
        self,
        *,
        rag: Optional[FilingRAG] = None,
        grader_llm: Optional[LLMProvider] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._rag = rag
        self._grader_llm = grader_llm
        base_cache = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
        self.cache_dir = base_cache / "eval_cache"

    # --- Golden-set loading --------------------------------------------------

    def load_golden_set(self, path: Optional[Path] = None) -> list[GoldenItem]:
        """Load and validate the YAML golden set.

        We require PyYAML for this; if it's not installed we raise an
        informative ImportError rather than silently returning [].
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "PyYAML is required to load the eval golden set. "
                "Install with `pip install pyyaml`."
            ) from exc

        if path is None:
            with _resources.as_file(
                _resources.files("filings_analyst.eval") / _DEFAULT_GOLDEN_FILENAME
            ) as p:
                raw = Path(p).read_text(encoding="utf-8")
        else:
            raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or []
        items: list[GoldenItem] = []
        for entry in data:
            items.append(
                GoldenItem(
                    id=entry["id"],
                    ticker=entry["ticker"],
                    accession_no=str(entry["accession_no"]),
                    question=entry["question"],
                    type=entry["type"],
                    expected_section=entry.get("expected_section"),
                    reference_answer=entry["reference_answer"],
                    scope=entry["scope"],
                    notes=entry.get("notes", "") or "",
                )
            )
        return items

    # --- Cache plumbing -----------------------------------------------------

    def _cache_path(self, item_id: str, provider_name: str) -> Path:
        safe = item_id.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}_{provider_name}.json"

    def _load_cached(self, item_id: str, provider_name: str) -> Optional[ItemResult]:
        path = self._cache_path(item_id, provider_name)
        if not path.exists():
            return None
        try:
            return ItemResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _store_cached(self, result: ItemResult, provider_name: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(result.item_id, provider_name)
        path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Main entry point ---------------------------------------------------

    def run(
        self,
        golden_set: list[GoldenItem],
        *,
        sample: Optional[int] = None,
        only_types: Optional[set[str]] = None,
        skip_cached: bool = True,
        progress: bool = True,
    ) -> EvalRunResult:
        rag = self._rag if self._rag is not None else FilingRAG()
        grader = self._grader_llm if self._grader_llm is not None else LLMProvider()
        provider_name = grader.provider if grader.available else "none"

        filtered = list(golden_set)
        if only_types:
            filtered = [g for g in filtered if g.type in only_types]
        if sample is not None and sample > 0:
            filtered = filtered[:sample]

        started_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        t0 = time.time()
        per_item: list[ItemResult] = []

        for i, item in enumerate(filtered, start=1):
            if progress:
                print(f"  [{i}/{len(filtered)}] {item.id} ({item.type}) {item.question[:70]}")
            if skip_cached:
                cached = self._load_cached(item.id, provider_name)
                if cached is not None and cached.metrics:
                    if progress:
                        print(f"      cache hit ({len(cached.metrics)} metrics)")
                    per_item.append(cached)
                    continue
            try:
                result = self._run_one(item, rag, grader)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc(limit=4)
                if progress:
                    print(f"      ERROR: {exc}")
                # Record an empty metrics result with the error captured.
                result = ItemResult(
                    item_id=item.id,
                    ticker=item.ticker,
                    type=item.type,
                    scope=item.scope,
                    question=item.question,
                    answer=None,
                    cited_chunks=[],
                    metrics={
                        name: MetricResult(
                            score=0.0,
                            reason=f"Item-level exception: {exc}",
                            raw_grade=None,
                        )
                        for name in _metrics.METRICS
                    },
                    error=f"{exc}\n{tb}",
                )
            self._store_cached(result, provider_name)
            per_item.append(result)

        duration = time.time() - t0
        aggregate = _mean_per_metric(per_item)

        return EvalRunResult(
            per_item_results=per_item,
            aggregate_metrics=aggregate,
            provider=provider_name,
            started_at=started_at,
            duration_s=duration,
            n_total=len(golden_set),
            n_run=len(filtered),
        )

    # --- Per-item execution -------------------------------------------------

    def _run_one(
        self,
        item: GoldenItem,
        rag: FilingRAG,
        grader: LLMProvider,
    ) -> ItemResult:
        rag_out = rag.ask_filing(
            item.question, accession_no=item.accession_no, ticker=item.ticker
        )
        answer = rag_out.get("answer")
        cited = rag_out.get("cited_chunks") or []

        metric_results: dict[str, MetricResult] = {}
        for name, fn in _metrics.METRICS.items():
            try:
                raw = fn(
                    item.question,
                    cited,
                    answer,
                    item.reference_answer,
                    item.scope,
                    llm=grader,
                )
                metric_results[name] = MetricResult(
                    score=float(raw.get("score", 0.0)),
                    reason=str(raw.get("reason", "")),
                    raw_grade=raw.get("raw_grade"),
                )
            except Exception as exc:  # noqa: BLE001
                metric_results[name] = MetricResult(
                    score=0.0,
                    reason=f"Metric {name} raised: {exc}",
                    raw_grade=None,
                )

        return ItemResult(
            item_id=item.id,
            ticker=item.ticker,
            type=item.type,
            scope=item.scope,
            question=item.question,
            answer=answer,
            cited_chunks=cited,
            metrics=metric_results,
            error=rag_out.get("error"),
        )


__all__ = (
    "EvalRunner",
    "EvalRunResult",
    "GoldenItem",
    "ItemResult",
    "MetricResult",
)
