# filings-analyst eval report

- Run started: `2026-05-20T11:59:33+00:00`
- Duration: `2318.8s`
- Provider: `claude_cli`
- Items run: `35` / `35` total
- Git SHA: `263711d`

## Aggregate metrics

| Metric | Score |
|--------|-------|
| faithfulness | 0.971 |
| answer_relevancy | 0.820 |
| context_precision | 0.524 |
| context_recall | 0.429 |
| refusal_correctness | 0.714 |

## Breakdowns

### Per-type breakdown

| Type | n | Mean (all metrics) | faithfulness | answer_relevancy | context_precision | context_recall | refusal_correctness |
|------|---|--------------------|--------------|------------------|-------------------|----------------|---------------------|
| easy | 15 | 0.716 | 0.933 | 0.915 | 0.500 | 0.433 | 0.800 |
| hard | 13 | 0.542 | 1.000 | 0.633 | 0.346 | 0.192 | 0.538 |
| out-of-scope | 7 | 0.917 | 1.000 | 0.964 | 0.905 | 0.857 | 0.857 |

### Per-ticker breakdown

| Ticker | n | Mean (all metrics) | faithfulness | answer_relevancy | context_precision | context_recall | refusal_correctness |
|--------|---|--------------------|--------------|------------------|-------------------|----------------|---------------------|
| AAPL | 7 | 0.739 | 1.000 | 0.836 | 0.500 | 0.643 | 0.714 |
| BAC | 7 | 0.721 | 1.000 | 0.916 | 0.476 | 0.500 | 0.714 |
| JPM | 7 | 0.744 | 1.000 | 0.889 | 0.619 | 0.357 | 0.857 |
| MSFT | 7 | 0.615 | 0.857 | 0.693 | 0.524 | 0.286 | 0.714 |
| XOM | 7 | 0.639 | 1.000 | 0.767 | 0.500 | 0.357 | 0.571 |

## Worst-3 items per metric


#### answer_relevancy

- `aapl-004` (score=0.00, type=hard, ticker=AAPL) — Inferred question: 'What is the capital of France?'. Grader similarity=0.00. The inferred question asks for a geographic fact (capital of France), while the original asks about Apple's framing of AI as both opportunity and risk — entirely unrelated topics.
  - Q: How does Apple frame the dual nature of artificial intelligence — both as an opportunity and as a risk to its business?
  - A: The provided excerpts do not contain that information.
- `msft-004` (score=0.00, type=hard, ticker=MSFT) — Grader returned unparseable similarity (inferred: "How does Microsoft's framing of AI differ between the Business/MD&A sections and the Risk Factors section of its filing?").
  - Q: How does Microsoft's framing of AI in the Business section differ from its framing of AI in the Risk Factors?
  - A: In the Business/MD&A excerpts, Microsoft frames AI as a growth driver and strategic opportunity — describing itself as "a technology company committed to making...
- `msft-005` (score=0.05, type=hard, ticker=MSFT) — Inferred question: 'What information is missing from the provided excerpts?'. Grader similarity=0.05. The original asks for specific disclosures about Microsoft's AI capex and expected returns, while the inferred question asks meta-information about gaps in excerpts — a fundamentally different ask.
  - Q: What does Microsoft disclose about the relationship between its AI capex and its expected return on those investments?
  - A: The provided excerpts do not contain that information.

#### context_precision

- `aapl-004` (score=0.00, type=hard, ticker=AAPL) — 0/6 chunks relevant. [1] not relevant; [2] not relevant; [3] not relevant; [4] not relevant; [5] not relevant; [6] not relevant
  - Q: How does Apple frame the dual nature of artificial intelligence — both as an opportunity and as a risk to its business?
  - A: The provided excerpts do not contain that information.
- `msft-004` (score=0.00, type=hard, ticker=MSFT) — 0/6 chunks relevant. [1] not relevant; [2] not relevant; [3] not relevant; [4] not relevant; [5] not relevant; [6] not relevant
  - Q: How does Microsoft's framing of AI in the Business section differ from its framing of AI in the Risk Factors?
  - A: In the Business/MD&A excerpts, Microsoft frames AI as a growth driver and strategic opportunity — describing itself as "a technology company committed to making...
- `xom-004` (score=0.00, type=hard, ticker=XOM) — 0/6 chunks relevant. [1] not relevant; [2] not relevant; [3] not relevant; [4] not relevant; [5] not relevant; [6] not relevant
  - Q: How does ExxonMobil's capital-allocation discussion balance investment in traditional oil and gas projects with investment in lower-carbon initiatives like carbon capture and lithium?
  - A: The provided excerpts do not contain that information.  The excerpts mention that "Capital investment guidance in lower-emission investments is based on our Cor...

#### context_recall

- `aapl-004` (score=0.00, type=hard, ticker=AAPL) — 0/1 reference sentences supported by retrieved chunks.
  - Q: How does Apple frame the dual nature of artificial intelligence — both as an opportunity and as a risk to its business?
  - A: The provided excerpts do not contain that information.
- `aapl-005` (score=0.00, type=hard, ticker=AAPL) — 0/1 reference sentences supported by retrieved chunks.
  - Q: How does Apple's discussion of regulatory and antitrust risk relate to its Services business?
  - A: Apple's regulatory and antitrust risk discussion connects to its Services business primarily through App Store-related actions and licensing arrangements:  - **...
- `bac-002` (score=0.00, type=easy, ticker=BAC) — 0/1 reference sentences supported by retrieved chunks.
  - Q: What does Bank of America disclose about its exposure to interest rate risk?
  - A: Bank of America discloses several aspects of its interest rate risk exposure:  - **Mortgage banking**: Interest rate risk and market risk "can be substantial in...

#### faithfulness

- `msft-001` (score=0.00, type=easy, ticker=MSFT) — Grader returned unparseable response.
  - Q: What does Microsoft describe as its operating segments?
  - A: Microsoft reports its financial performance using three operating segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing ...
- `aapl-001` (score=1.00, type=easy, ticker=AAPL) — 10/10 claims supported by context.
  - Q: What does Apple identify as risks related to its supply chain and reliance on outsourcing partners?
  - A: Apple identifies several supply chain and outsourcing risks:  - **Geographic concentration of manufacturing**: Substantially all hardware products are manufactu...
- `aapl-002` (score=1.00, type=easy, ticker=AAPL) — 16/16 claims supported by context.
  - Q: What does Apple identify as its primary product categories?
  - A: Apple identifies its primary product categories as iPhone (smartphones based on iOS), Mac (personal computers based on macOS), iPad (tablets based on iPadOS), a...

#### refusal_correctness

- `aapl-003` (score=0.00, type=easy, ticker=AAPL) — In-scope question but system refused — either retrieval failed or the model was overly cautious.
  - Q: What does Apple disclose about intellectual property risks?
  - A: Apple discloses several intellectual property (IP) risks in its 2025 10-K:  - **Reliance on third-party IP licenses**: Many of Apple's products and services inc...
- `aapl-004` (score=0.00, type=hard, ticker=AAPL) — In-scope question but system refused — either retrieval failed or the model was overly cautious.
  - Q: How does Apple frame the dual nature of artificial intelligence — both as an opportunity and as a risk to its business?
  - A: The provided excerpts do not contain that information.
- `bac-002` (score=0.00, type=easy, ticker=BAC) — In-scope question but system refused — either retrieval failed or the model was overly cautious.
  - Q: What does Bank of America disclose about its exposure to interest rate risk?
  - A: Bank of America discloses several aspects of its interest rate risk exposure:  - **Mortgage banking**: Interest rate risk and market risk "can be substantial in...

## Methodology notes

- Golden set is hand-curated, 35 items across AAPL / MSFT / JPM / BAC / XOM.
- `refusal_correctness` is deterministic (substring + citation regex). The other four metrics route grader prompts through `LLMProvider`.
- For out-of-scope items the system *should* refuse; faithfulness,   answer_relevancy, context_precision, and context_recall short-circuit   to 1.0 when the refusal is detected, on the principle that a correct   refusal is not a retrieval failure.
- Re-run with: `filings-analyst eval`.
