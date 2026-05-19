# filings-analyst eval report

- Run started: `2026-05-19T21:34:03+00:00`
- Duration: `612.9s`
- Provider: `claude_cli`
- Items run: `10` / `10` total
- Git SHA: `b68058e`

## Aggregate metrics

| Metric | Score |
|--------|-------|
| faithfulness | 0.992 |
| answer_relevancy | 0.957 |
| context_precision | 0.650 |
| context_recall | 0.500 |
| refusal_correctness | 1.000 |

## Breakdowns

### Per-type breakdown

| Type | n | Mean (all metrics) | faithfulness | answer_relevancy | context_precision | context_recall | refusal_correctness |
|------|---|--------------------|--------------|------------------|-------------------|----------------|---------------------|
| easy | 4 | 0.808 | 1.000 | 0.955 | 0.583 | 0.500 | 1.000 |
| hard | 3 | 0.656 | 0.974 | 0.917 | 0.389 | 0.000 | 1.000 |
| out-of-scope | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Per-ticker breakdown

| Ticker | n | Mean (all metrics) | faithfulness | answer_relevancy | context_precision | context_recall | refusal_correctness |
|--------|---|--------------------|--------------|------------------|-------------------|----------------|---------------------|
| AAPL | 3 | 0.845 | 0.974 | 0.973 | 0.611 | 0.667 | 1.000 |
| BAC | 1 | 0.890 | 1.000 | 0.950 | 0.500 | 1.000 | 1.000 |
| JPM | 3 | 0.801 | 1.000 | 0.950 | 0.722 | 0.333 | 1.000 |
| MSFT | 2 | 0.685 | 1.000 | 0.925 | 0.500 | 0.000 | 1.000 |
| XOM | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Worst-3 items per metric


#### answer_relevancy

- `jpm-005` (score=0.90, type=hard, ticker=JPM) — Inferred question: 'How does JPMorgan address operational risk, particularly regarding AI, model governance, third-party vendors, and conduct?'. Grader similarity=0.90. Both questions ask about JPMorgan's approach to operational risk with specific focus on AI and third-party vendors. The inferred question adds model governance and conduct as additional sub-topics, slightly broadening the scope, but the core information requested is essentially the same.
  - Q: How does JPMorgan describe its approach to operational risk, including risks from artificial intelligence and third-party vendors?
  - A: JPMorgan describes operational risk as encompassing several subcategories — including conduct risk, estimations and model risk, and cybersecurity risk — each wi...
- `msft-004` (score=0.90, type=hard, ticker=MSFT) — Inferred question: "How does Microsoft's framing of AI differ between the Business/MD&A sections and the Risk Factors section of its filing?". Grader similarity=0.90. Both questions ask how Microsoft frames AI differently between forward-looking/operational sections and Risk Factors. The inferred question adds MD&A alongside Business, slightly broadening the scope, but the core comparison (positive/opportunity framing vs. risk framing of AI) is essentially the same.
  - Q: How does Microsoft's framing of AI in the Business section differ from its framing of AI in the Risk Factors?
  - A: In the Business/MD&A framing, AI is presented as a growth driver and strategic opportunity — Microsoft describes itself as "a technology company committed to ma...
- `aapl-005` (score=0.95, type=hard, ticker=AAPL) — Inferred question: "What does Apple's 10-K say about regulatory and antitrust risks affecting its Services business?". Grader similarity=0.95. Both questions ask about regulatory and antitrust risks as they relate to Apple's Services business. The inferred question explicitly scopes to the 10-K, but this is the implicit source for the original question as well, so they seek essentially the same information.
  - Q: How does Apple's discussion of regulatory and antitrust risk relate to its Services business?
  - A: Apple's 10-K describes regulatory and antitrust pressures that directly affect its Services business, particularly the App Store. In the U.S., Apple is under a ...

#### context_precision

- `msft-004` (score=0.17, type=hard, ticker=MSFT) — 1/6 chunks relevant. [1] not relevant; [2] relevant; [3] not relevant; [4] not relevant; [5] not relevant; [6] not relevant
  - Q: How does Microsoft's framing of AI in the Business section differ from its framing of AI in the Risk Factors?
  - A: In the Business/MD&A framing, AI is presented as a growth driver and strategic opportunity — Microsoft describes itself as "a technology company committed to ma...
- `aapl-005` (score=0.33, type=hard, ticker=AAPL) — 2/6 chunks relevant. [1] not relevant; [2] relevant; [3] not relevant; [4] not relevant; [5] relevant; [6] not relevant
  - Q: How does Apple's discussion of regulatory and antitrust risk relate to its Services business?
  - A: Apple's 10-K describes regulatory and antitrust pressures that directly affect its Services business, particularly the App Store. In the U.S., Apple is under a ...
- `aapl-001` (score=0.50, type=easy, ticker=AAPL) — 3/6 chunks relevant. [1] relevant; [2] relevant; [3] not relevant; [4] not relevant; [5] relevant; [6] not relevant
  - Q: What does Apple identify as risks related to its supply chain and reliance on outsourcing partners?
  - A: Apple identifies several supply-chain and outsourcing risks:  - **Geographic concentration of manufacturing.** Substantially all hardware products are manufactu...

#### context_recall

- `aapl-005` (score=0.00, type=hard, ticker=AAPL) — 0/1 reference sentences supported by retrieved chunks.
  - Q: How does Apple's discussion of regulatory and antitrust risk relate to its Services business?
  - A: Apple's 10-K describes regulatory and antitrust pressures that directly affect its Services business, particularly the App Store. In the U.S., Apple is under a ...
- `jpm-001` (score=0.00, type=easy, ticker=JPM) — 0/1 reference sentences supported by retrieved chunks.
  - Q: What does JPMorgan Chase describe as its primary business segments?
  - A: JPMorgan Chase describes three reportable business segments for management reporting purposes: Consumer & Community Banking (CCB), Commercial & Investment Bank ...
- `jpm-005` (score=0.00, type=hard, ticker=JPM) — 0/2 reference sentences supported by retrieved chunks.
  - Q: How does JPMorgan describe its approach to operational risk, including risks from artificial intelligence and third-party vendors?
  - A: JPMorgan describes operational risk as encompassing several subcategories — including conduct risk, estimations and model risk, and cybersecurity risk — each wi...

#### faithfulness

- `aapl-005` (score=0.92, type=hard, ticker=AAPL) — 12/13 claims supported by context.
  - Q: How does Apple's discussion of regulatory and antitrust risk relate to its Services business?
  - A: Apple's 10-K describes regulatory and antitrust pressures that directly affect its Services business, particularly the App Store. In the U.S., Apple is under a ...
- `aapl-001` (score=1.00, type=easy, ticker=AAPL) — 13/13 claims supported by context.
  - Q: What does Apple identify as risks related to its supply chain and reliance on outsourcing partners?
  - A: Apple identifies several supply-chain and outsourcing risks:  - **Geographic concentration of manufacturing.** Substantially all hardware products are manufactu...
- `aapl-006` (score=1.00, type=out-of-scope, ticker=AAPL) — Out-of-scope refusal — no claims to verify.
  - Q: What was the salary of Apple's Chief Executive Officer for fiscal year 2025?
  - A: The provided excerpts do not contain that information.

#### refusal_correctness

- `aapl-001` (score=1.00, type=easy, ticker=AAPL) — In-scope question: system answered and cited.
  - Q: What does Apple identify as risks related to its supply chain and reliance on outsourcing partners?
  - A: Apple identifies several supply-chain and outsourcing risks:  - **Geographic concentration of manufacturing.** Substantially all hardware products are manufactu...
- `aapl-005` (score=1.00, type=hard, ticker=AAPL) — In-scope question: system answered and cited.
  - Q: How does Apple's discussion of regulatory and antitrust risk relate to its Services business?
  - A: Apple's 10-K describes regulatory and antitrust pressures that directly affect its Services business, particularly the App Store. In the U.S., Apple is under a ...
- `aapl-006` (score=1.00, type=out-of-scope, ticker=AAPL) — Out-of-scope question: system correctly refused.
  - Q: What was the salary of Apple's Chief Executive Officer for fiscal year 2025?
  - A: The provided excerpts do not contain that information.

## Methodology notes

- Golden set is hand-curated, 35 items across AAPL / MSFT / JPM / BAC / XOM.
- `refusal_correctness` is deterministic (substring + citation regex). The other four metrics route grader prompts through `LLMProvider`.
- For out-of-scope items the system *should* refuse; faithfulness,   answer_relevancy, context_precision, and context_recall short-circuit   to 1.0 when the refusal is detected, on the principle that a correct   refusal is not a retrieval failure.
- Re-run with: `filings-analyst eval`.
