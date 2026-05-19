"""Five evaluation metrics for the filings-analyst RAG pipeline.

Four of these (faithfulness, answer_relevancy, context_precision,
context_recall) are ragas-style. The fifth (refusal_correctness) is
hand-rolled and entirely deterministic — it's the one that measures the
behavior most RAG portfolios skip: correctly refusing to answer
questions whose answer isn't in the corpus.

We deliberately do NOT depend on the ragas library at import time:

* ragas pulls in a hairy dependency tree (langchain, datasets, etc.)
  that we don't want to gate the package install on.
* The portfolio purpose of this file is to *demonstrate* that we know
  what each ragas metric is doing — implementing the same idea on top
  of our existing ``LLMProvider`` is more transparent for a recruiter
  reading the source than calling an opaque library.

If ragas is installed, ``METRICS`` will be no different — we just route
the same grader prompts through ``LLMProvider`` either way. A future
version could switch on ``import ragas`` and delegate; this is noted
in the README evaluation section.

Each metric is a callable with the same signature:

    def metric(question, retrieved_chunks, answer, reference_answer, scope) -> dict

Return shape::

    {"score": float, "reason": str, "raw_grade": str | None}

The ``reason`` field always explains WHY the score came out the way it
did, so a human auditor reading the worst-items section of the report
can tell whether the metric is fair or whether the grading itself is
the bug.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from ..providers import LLMProvider


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _chunks_to_text(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as a numbered block for grader prompts."""
    if not retrieved_chunks:
        return "(no chunks retrieved)"
    parts: list[str] = []
    for i, c in enumerate(retrieved_chunks, start=1):
        section = c.get("section", "?")
        idx = c.get("chunk_idx", "?")
        text = (c.get("text") or "").strip()
        parts.append(f"[{i}] Section: {section} §{idx}\n{text}")
    return "\n\n".join(parts)


def _extract_json(blob: Optional[str]) -> Optional[dict]:
    """Parse the first JSON object out of an LLM response.

    LLM graders are notorious for wrapping JSON in prose. We strip code
    fences and search for the first balanced JSON object. Returns
    ``None`` if nothing parseable is found.
    """
    if not blob:
        return None
    text = blob.strip()
    # Strip fenced code blocks.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Find the first { ... } block via simple bracket counting.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter — good enough for grading prompts."""
    if not text:
        return []
    # Split on ., ?, ! followed by whitespace; keep non-empty pieces.
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


# ---------------------------------------------------------------------------
# Refusal language — the one metric that does NOT need an LLM
# ---------------------------------------------------------------------------

# Phrases that indicate the system declined to answer because the
# answer was not in the retrieved context. Matched case-insensitively
# as substrings (not regexes — we want zero false negatives from
# minor wording variation).
_REFUSAL_PHRASES = (
    "do not contain that information",
    "does not contain that information",
    "not in the provided",
    "not in the filing",
    "not contained in the",
    "cannot answer based on",
    "cannot be answered from",
    "i do not have information",
    "i don't have information",
    "i don't see this in",
    "i do not see this in",
    "out of scope",
    "the provided excerpts do not",
    "no information about",
    "no relevant information",
)


def _looks_like_refusal(answer: Optional[str]) -> bool:
    if not answer:
        # No answer at all counts as a refusal — the system didn't
        # hallucinate content for a question it couldn't answer.
        return True
    low = answer.lower()
    return any(p in low for p in _REFUSAL_PHRASES)


_CITATION_PATTERNS = (
    re.compile(r"\[[^\]]+§\s*\d+\]"),         # [Risk Factors §3]
    re.compile(r"\[\s*\d+\s*\]"),               # [1]
    re.compile(r"\[[A-Z]{2,6}\s+[^\]]+\]"),    # [AAPL Risk Factors §3]
)


def _has_citation(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return any(p.search(answer) for p in _CITATION_PATTERNS)


# ---------------------------------------------------------------------------
# Metric: refusal_correctness (hand-rolled, deterministic)
# ---------------------------------------------------------------------------


def refusal_correctness(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: Optional[str],
    reference_answer: str,
    scope: str,
    *,
    llm: Optional[LLMProvider] = None,
) -> dict:
    """Was the refuse/answer decision correct?

    For out-of-scope items, the system should have refused — score 1.0
    if refusal language is detected, else 0.0.

    For in-scope items, the system should have NOT refused AND should
    have produced at least one citation — score 1.0 if both, else 0.0.
    """
    refused = _looks_like_refusal(answer)
    if scope == "out-of-scope":
        if refused:
            return {
                "score": 1.0,
                "reason": "Out-of-scope question: system correctly refused.",
                "raw_grade": None,
            }
        return {
            "score": 0.0,
            "reason": (
                "Out-of-scope question but system produced an answer "
                "instead of refusing — possible hallucination."
            ),
            "raw_grade": None,
        }
    # in-scope
    if refused:
        return {
            "score": 0.0,
            "reason": (
                "In-scope question but system refused — either retrieval "
                "failed or the model was overly cautious."
            ),
            "raw_grade": None,
        }
    if not _has_citation(answer):
        return {
            "score": 0.0,
            "reason": (
                "In-scope question, system answered without any citation "
                "in [Section §idx] or [TICKER ...] form."
            ),
            "raw_grade": None,
        }
    return {
        "score": 1.0,
        "reason": "In-scope question: system answered and cited.",
        "raw_grade": None,
    }


# ---------------------------------------------------------------------------
# Metric: faithfulness
# ---------------------------------------------------------------------------


_FAITHFULNESS_PROMPT = """You are grading whether an answer is faithful to a set of retrieved context excerpts.

Step 1: Extract each distinct factual claim made in the ANSWER.
Step 2: For each claim, decide whether it is supported by at least one of the CONTEXT excerpts.
Step 3: Return STRICT JSON only, of the form:
{{"claims": ["claim 1", "claim 2", ...], "supported": [true, false, ...]}}

The lists must be the same length. Do not include any prose outside the JSON.

=== Context ===
{context}

=== Answer ===
{answer}

=== JSON ===
"""


def faithfulness(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: Optional[str],
    reference_answer: str,
    scope: str,
    *,
    llm: Optional[LLMProvider] = None,
) -> dict:
    """Per-claim support check.

    score = supported_claims / total_claims.

    For out-of-scope items that the system correctly refused, we
    short-circuit to 1.0 — there are no claims to be unfaithful to.
    """
    if not answer:
        return {
            "score": 1.0 if scope == "out-of-scope" else 0.0,
            "reason": "No answer produced — vacuously faithful only for out-of-scope.",
            "raw_grade": None,
        }
    if scope == "out-of-scope" and _looks_like_refusal(answer):
        return {
            "score": 1.0,
            "reason": "Out-of-scope refusal — no claims to verify.",
            "raw_grade": None,
        }

    grader = llm or LLMProvider()
    if not grader.available:
        return {
            "score": 0.0,
            "reason": "No grader LLM available.",
            "raw_grade": None,
        }
    prompt = _FAITHFULNESS_PROMPT.format(
        context=_chunks_to_text(retrieved_chunks),
        answer=answer,
    )
    raw = grader.generate(prompt, max_tokens=800)
    parsed = _extract_json(raw)
    if not parsed:
        return {
            "score": 0.0,
            "reason": "Grader returned unparseable response.",
            "raw_grade": raw,
        }
    claims = parsed.get("claims") or []
    supported = parsed.get("supported") or []
    if not claims or len(claims) != len(supported):
        return {
            "score": 0.0,
            "reason": "Grader returned malformed claims/supported lists.",
            "raw_grade": raw,
        }
    supported_count = sum(1 for s in supported if bool(s))
    score = supported_count / len(claims)
    return {
        "score": _clamp01(score),
        "reason": (
            f"{supported_count}/{len(claims)} claims supported by context."
        ),
        "raw_grade": raw,
    }


# ---------------------------------------------------------------------------
# Metric: answer_relevancy
# ---------------------------------------------------------------------------


_RELEVANCY_INFER_PROMPT = """Read the answer below and write a single short question that this answer is addressing. Output only the question text — no quotes, no labels, no commentary.

=== Answer ===
{answer}

=== Inferred question ==="""


_RELEVANCY_COMPARE_PROMPT = """You are scoring how closely two questions are asking for the same information.

Original question: {original}
Inferred question: {inferred}

Return STRICT JSON only of the form:
{{"score": 0.0 to 1.0, "reason": "..."}}

1.0 means the inferred question is asking for essentially the same information as the original. 0.0 means completely different.

=== JSON ==="""


def answer_relevancy(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: Optional[str],
    reference_answer: str,
    scope: str,
    *,
    llm: Optional[LLMProvider] = None,
) -> dict:
    """Does the answer actually address the question?

    Implementation: ask grader to infer what question the answer was
    answering, then ask grader how similar that is to the actual
    question. Score = similarity.

    For out-of-scope items where the system correctly refused, score
    1.0 — refusing is the relevant response.
    """
    if not answer:
        return {
            "score": 1.0 if scope == "out-of-scope" else 0.0,
            "reason": "No answer produced.",
            "raw_grade": None,
        }
    if scope == "out-of-scope" and _looks_like_refusal(answer):
        return {
            "score": 1.0,
            "reason": "Out-of-scope refusal is the relevant response.",
            "raw_grade": None,
        }

    grader = llm or LLMProvider()
    if not grader.available:
        return {
            "score": 0.0,
            "reason": "No grader LLM available.",
            "raw_grade": None,
        }
    inferred = grader.generate(
        _RELEVANCY_INFER_PROMPT.format(answer=answer), max_tokens=200
    )
    if not inferred:
        return {
            "score": 0.0,
            "reason": "Grader could not infer a question from the answer.",
            "raw_grade": None,
        }
    inferred = inferred.strip().splitlines()[0].strip()
    raw = grader.generate(
        _RELEVANCY_COMPARE_PROMPT.format(original=question, inferred=inferred),
        max_tokens=200,
    )
    parsed = _extract_json(raw)
    if not parsed or "score" not in parsed:
        return {
            "score": 0.0,
            "reason": f"Grader returned unparseable similarity (inferred: {inferred!r}).",
            "raw_grade": raw,
        }
    score = _clamp01(parsed.get("score", 0.0))
    return {
        "score": score,
        "reason": (
            f"Inferred question: {inferred!r}. Grader similarity={score:.2f}. "
            f"{parsed.get('reason', '')}".strip()
        ),
        "raw_grade": raw,
    }


# ---------------------------------------------------------------------------
# Metric: context_precision
# ---------------------------------------------------------------------------


_PRECISION_PROMPT = """Decide whether the following CHUNK is relevant to answering the QUESTION (use the REFERENCE ANSWER as a hint about what a good answer looks like).

Return STRICT JSON only of the form:
{{"relevant": true | false, "reason": "..."}}

=== Question ===
{question}

=== Reference answer ===
{reference}

=== Chunk ===
{chunk}

=== JSON ==="""


def context_precision(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: Optional[str],
    reference_answer: str,
    scope: str,
    *,
    llm: Optional[LLMProvider] = None,
) -> dict:
    """What fraction of retrieved chunks were relevant?

    For out-of-scope items, the "correct" retrieval is one that
    returns nothing useful, so we score precision as 1.0 when the
    system correctly refused (regardless of retrieved content) — the
    refusal already captured that the retrieval didn't bear fruit.
    Scoring retrieval as low here would double-penalise refusal cases.
    """
    if scope == "out-of-scope":
        if _looks_like_refusal(answer):
            return {
                "score": 1.0,
                "reason": "Out-of-scope refusal — retrieval precision not penalised.",
                "raw_grade": None,
            }
    if not retrieved_chunks:
        return {
            "score": 0.0,
            "reason": "No chunks retrieved.",
            "raw_grade": None,
        }

    grader = llm or LLMProvider()
    if not grader.available:
        return {
            "score": 0.0,
            "reason": "No grader LLM available.",
            "raw_grade": None,
        }
    relevant_count = 0
    per_chunk_reasons: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        prompt = _PRECISION_PROMPT.format(
            question=question,
            reference=reference_answer,
            chunk=(chunk.get("text") or "").strip(),
        )
        raw = grader.generate(prompt, max_tokens=200)
        parsed = _extract_json(raw)
        if parsed and parsed.get("relevant") is True:
            relevant_count += 1
            per_chunk_reasons.append(f"[{i}] relevant")
        else:
            per_chunk_reasons.append(f"[{i}] not relevant")
    score = relevant_count / len(retrieved_chunks)
    return {
        "score": _clamp01(score),
        "reason": (
            f"{relevant_count}/{len(retrieved_chunks)} chunks relevant. "
            + "; ".join(per_chunk_reasons)
        ),
        "raw_grade": None,
    }


# ---------------------------------------------------------------------------
# Metric: context_recall
# ---------------------------------------------------------------------------


_RECALL_PROMPT = """Decide whether the SENTENCE (taken from a reference answer) is supported by at least one of the retrieved CONTEXT excerpts.

Return STRICT JSON only of the form:
{{"supported": true | false, "reason": "..."}}

=== Context ===
{context}

=== Sentence ===
{sentence}

=== JSON ==="""


def context_recall(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: Optional[str],
    reference_answer: str,
    scope: str,
    *,
    llm: Optional[LLMProvider] = None,
) -> dict:
    """For each sentence in the reference answer, is it supported by retrieved chunks?

    For out-of-scope items, recall is not meaningful (there's no
    information that *should* have been retrieved). We short-circuit
    to 1.0 when the system correctly refused.
    """
    if scope == "out-of-scope" and _looks_like_refusal(answer):
        return {
            "score": 1.0,
            "reason": "Out-of-scope refusal — recall not applicable.",
            "raw_grade": None,
        }
    sentences = _split_sentences(reference_answer)
    if not sentences:
        return {
            "score": 0.0,
            "reason": "Reference answer empty.",
            "raw_grade": None,
        }
    if not retrieved_chunks:
        return {
            "score": 0.0,
            "reason": "No chunks retrieved.",
            "raw_grade": None,
        }

    grader = llm or LLMProvider()
    if not grader.available:
        return {
            "score": 0.0,
            "reason": "No grader LLM available.",
            "raw_grade": None,
        }
    context_text = _chunks_to_text(retrieved_chunks)
    supported_count = 0
    for sentence in sentences:
        prompt = _RECALL_PROMPT.format(context=context_text, sentence=sentence)
        raw = grader.generate(prompt, max_tokens=200)
        parsed = _extract_json(raw)
        if parsed and parsed.get("supported") is True:
            supported_count += 1
    score = supported_count / len(sentences)
    return {
        "score": _clamp01(score),
        "reason": (
            f"{supported_count}/{len(sentences)} reference sentences supported "
            "by retrieved chunks."
        ),
        "raw_grade": None,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


MetricFn = Callable[..., dict]

METRICS: dict[str, MetricFn] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "refusal_correctness": refusal_correctness,
}


__all__ = (
    "METRICS",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "refusal_correctness",
)
