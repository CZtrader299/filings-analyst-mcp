"""Formal evaluation harness for the filings-analyst RAG pipeline.

Modules:

* :mod:`filings_analyst.eval.metrics` — the five grading functions.
* :mod:`filings_analyst.eval.runner` — orchestrator with disk cache.
* :mod:`filings_analyst.eval.report` — markdown report generators.

The golden set itself lives at :file:`golden_set.yaml` alongside this
package — hand-curated, NOT LLM-generated, because LLM-generated golden
sets defeat the purpose of measuring whether the LLM is correct.
"""

from .runner import EvalRunner, EvalRunResult, GoldenItem, ItemResult

__all__ = ("EvalRunner", "EvalRunResult", "GoldenItem", "ItemResult")
