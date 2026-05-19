# Eval reports

Markdown reports produced by `filings-analyst eval`. Each file is a
full snapshot of one run: aggregate metric table, per-type and
per-ticker breakdowns, and the worst-3 items per metric so a reader
can see where the system fails.

## How to read these

The five metrics:

- **faithfulness** — fraction of claims in the answer that are
  supported by the retrieved chunks. Low = hallucination.
- **answer_relevancy** — semantic similarity between the actual
  question and the question the answer *appears* to be addressing.
  Low = off-topic answer.
- **context_precision** — fraction of retrieved chunks that are
  actually relevant. Low = noisy retrieval.
- **context_recall** — fraction of reference-answer sentences that
  are supported by retrieved chunks. Low = retrieval misses important
  information.
- **refusal_correctness** — hand-rolled, deterministic. Did the
  system correctly refuse out-of-scope questions and correctly answer
  (with a citation) in-scope ones?

For out-of-scope items, the four ragas-style metrics short-circuit to
1.0 when refusal is detected. We do *not* want to double-penalise a
correct refusal by also marking its retrieval as "low precision" —
the refusal already captured the fact that retrieval found nothing
useful.

## Reproducing

```
filings-analyst ingest --tickers AAPL,MSFT,JPM,BAC,XOM
filings-analyst eval --output eval_reports/<date>_eval.md
```

The harness disk-caches per-item grades under
`~/.filings_analyst_cache/eval_cache/` keyed by `(item_id,
provider)`, so re-runs only re-grade items whose system-under-test
changed.
