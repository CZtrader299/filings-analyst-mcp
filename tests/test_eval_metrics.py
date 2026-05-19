"""Tests for the five eval metrics.

All LLM calls are mocked. The deterministic ``refusal_correctness``
metric needs no mocking.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from filings_analyst.eval import metrics as M


def _make_llm(responses):
    """Build a MagicMock LLMProvider that returns the queued responses in order."""
    llm = MagicMock()
    llm.available = True
    llm.provider = "anthropic_api"
    llm.generate = MagicMock(side_effect=list(responses))
    return llm


# ---------------------------------------------------------------------------
# refusal_correctness — deterministic
# ---------------------------------------------------------------------------


def test_refusal_correctness_oos_refusal_scores_one():
    out = M.refusal_correctness(
        question="What is X?",
        retrieved_chunks=[],
        answer="The provided excerpts do not contain that information.",
        reference_answer="OOS",
        scope="out-of-scope",
    )
    assert out["score"] == 1.0


def test_refusal_correctness_oos_answer_scores_zero():
    out = M.refusal_correctness(
        question="What is X?",
        retrieved_chunks=[{"section": "Risk Factors", "chunk_idx": 0, "text": "..."}],
        answer="X is 42. [Risk Factors §0]",
        reference_answer="OOS",
        scope="out-of-scope",
    )
    assert out["score"] == 0.0


def test_refusal_correctness_inscope_answered_with_citation_scores_one():
    out = M.refusal_correctness(
        question="What is X?",
        retrieved_chunks=[{"section": "Business", "chunk_idx": 1, "text": "..."}],
        answer="X is Apple's strategy. [Business §1]",
        reference_answer="X is Apple's strategy.",
        scope="in-scope",
    )
    assert out["score"] == 1.0


def test_refusal_correctness_inscope_refused_scores_zero():
    out = M.refusal_correctness(
        question="What is X?",
        retrieved_chunks=[{"section": "Business", "chunk_idx": 1, "text": "..."}],
        answer="The provided excerpts do not contain that information.",
        reference_answer="X is Apple's strategy.",
        scope="in-scope",
    )
    assert out["score"] == 0.0


def test_refusal_correctness_inscope_answered_no_citation_scores_zero():
    out = M.refusal_correctness(
        question="What is X?",
        retrieved_chunks=[{"section": "Business", "chunk_idx": 1, "text": "..."}],
        answer="X is Apple's strategy.",
        reference_answer="X is Apple's strategy.",
        scope="in-scope",
    )
    assert out["score"] == 0.0


# ---------------------------------------------------------------------------
# faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_all_supported():
    llm = _make_llm([
        json.dumps({"claims": ["A", "B"], "supported": [True, True]})
    ])
    out = M.faithfulness(
        question="Q?",
        retrieved_chunks=[{"section": "S", "chunk_idx": 0, "text": "A and B"}],
        answer="A and B.",
        reference_answer="A and B.",
        scope="in-scope",
        llm=llm,
    )
    assert out["score"] == 1.0


def test_faithfulness_partial_support():
    llm = _make_llm([
        json.dumps({"claims": ["A", "B", "C"], "supported": [True, False, True]})
    ])
    out = M.faithfulness(
        "Q?", [{"section": "S", "chunk_idx": 0, "text": "..."}], "A and B and C.", "ref", "in-scope", llm=llm
    )
    assert abs(out["score"] - 2 / 3) < 1e-6


def test_faithfulness_unparseable_grader_response():
    llm = _make_llm(["not even close to JSON"])
    out = M.faithfulness(
        "Q?", [{"section": "S", "chunk_idx": 0, "text": "..."}], "Some answer.", "ref", "in-scope", llm=llm
    )
    assert out["score"] == 0.0


def test_faithfulness_oos_refusal_short_circuits():
    llm = _make_llm([])  # would error if called
    out = M.faithfulness(
        "Q?",
        [],
        "The provided excerpts do not contain that information.",
        "ref",
        "out-of-scope",
        llm=llm,
    )
    assert out["score"] == 1.0
    llm.generate.assert_not_called()


# ---------------------------------------------------------------------------
# answer_relevancy
# ---------------------------------------------------------------------------


def test_answer_relevancy_high_similarity():
    llm = _make_llm([
        "What are Apple's supply chain risks?",
        json.dumps({"score": 0.95, "reason": "Both ask about supply chain risks."}),
    ])
    out = M.answer_relevancy(
        "What are Apple's supply chain risks?",
        [],
        "Apple's supply chain risks include concentration in Asia.",
        "ref",
        "in-scope",
        llm=llm,
    )
    assert out["score"] >= 0.9


def test_answer_relevancy_low_similarity():
    llm = _make_llm([
        "What is the weather today?",
        json.dumps({"score": 0.05, "reason": "Different topics."}),
    ])
    out = M.answer_relevancy(
        "What are Apple's supply chain risks?", [], "It is sunny.", "ref", "in-scope", llm=llm
    )
    assert out["score"] <= 0.1


def test_answer_relevancy_oos_refusal_short_circuits():
    llm = _make_llm([])
    out = M.answer_relevancy(
        "Q?", [], "Not in the provided context.", "ref", "out-of-scope", llm=llm
    )
    assert out["score"] == 1.0


# ---------------------------------------------------------------------------
# context_precision
# ---------------------------------------------------------------------------


def test_context_precision_all_relevant():
    llm = _make_llm([
        json.dumps({"relevant": True, "reason": "yes"}),
        json.dumps({"relevant": True, "reason": "yes"}),
    ])
    chunks = [
        {"section": "S", "chunk_idx": 0, "text": "x"},
        {"section": "S", "chunk_idx": 1, "text": "y"},
    ]
    out = M.context_precision("Q?", chunks, "ans", "ref", "in-scope", llm=llm)
    assert out["score"] == 1.0


def test_context_precision_mixed_relevance():
    llm = _make_llm([
        json.dumps({"relevant": True}),
        json.dumps({"relevant": False}),
        json.dumps({"relevant": True}),
    ])
    chunks = [
        {"section": "S", "chunk_idx": i, "text": "x"} for i in range(3)
    ]
    out = M.context_precision("Q?", chunks, "ans", "ref", "in-scope", llm=llm)
    assert abs(out["score"] - 2 / 3) < 1e-6


def test_context_precision_oos_refusal_short_circuits():
    llm = _make_llm([])
    out = M.context_precision(
        "Q?",
        [{"section": "S", "chunk_idx": 0, "text": "x"}],
        "Not in the provided context.",
        "ref",
        "out-of-scope",
        llm=llm,
    )
    assert out["score"] == 1.0
    llm.generate.assert_not_called()


# ---------------------------------------------------------------------------
# context_recall
# ---------------------------------------------------------------------------


def test_context_recall_all_supported():
    llm = _make_llm([
        json.dumps({"supported": True}),
        json.dumps({"supported": True}),
    ])
    out = M.context_recall(
        "Q?",
        [{"section": "S", "chunk_idx": 0, "text": "x"}],
        "ans",
        "First sentence. Second sentence.",
        "in-scope",
        llm=llm,
    )
    assert out["score"] == 1.0


def test_context_recall_partial():
    llm = _make_llm([
        json.dumps({"supported": True}),
        json.dumps({"supported": False}),
    ])
    out = M.context_recall(
        "Q?",
        [{"section": "S", "chunk_idx": 0, "text": "x"}],
        "ans",
        "First. Second.",
        "in-scope",
        llm=llm,
    )
    assert out["score"] == 0.5


def test_context_recall_oos_short_circuit():
    llm = _make_llm([])
    out = M.context_recall(
        "Q?",
        [{"section": "S", "chunk_idx": 0, "text": "x"}],
        "Not in the provided context.",
        "ref",
        "out-of-scope",
        llm=llm,
    )
    assert out["score"] == 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_extract_json_strips_fences():
    out = M._extract_json("```json\n{\"a\": 1}\n```")
    assert out == {"a": 1}


def test_extract_json_handles_prose_around_json():
    out = M._extract_json("Sure! Here is the JSON: {\"a\": 1, \"b\": [2, 3]} -- enjoy.")
    assert out == {"a": 1, "b": [2, 3]}


def test_extract_json_returns_none_on_garbage():
    assert M._extract_json("no JSON anywhere") is None


def test_has_citation_detects_section_idx():
    assert M._has_citation("Apple discusses risk. [Risk Factors §3]")
    assert M._has_citation("See [1] for details.")
    assert M._has_citation("Multi-filing [AAPL Risk Factors §3].")


def test_has_citation_false_without_citation():
    assert not M._has_citation("This is just prose with no brackets.")


def test_looks_like_refusal_catches_common_phrasings():
    assert M._looks_like_refusal("The provided excerpts do not contain that information.")
    assert M._looks_like_refusal("This question is out of scope.")
    assert M._looks_like_refusal("I do not have information about that.")
    assert not M._looks_like_refusal("Apple discusses risk factors at length.")
