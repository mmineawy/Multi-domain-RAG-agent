# Design Document

## 1. Design decisions
- **One agent, config-driven domains.** All domain behavior lives in `domains/<name>.yaml` and `knowledge_base/<name>/`. The Python code contains no domain-specific logic, which is the adaptation requirement of the assignment.
- **One vector collection per domain.** Prevents cross-domain leakage (a marketing question can never retrieve medical text).
- **Structured JSON output** (summary, key points, recommendations, risks, sources, self-confidence) so the UI, guardrails and evaluation can consume answers reliably.
- **Layered defence against hallucination** rather than relying on the prompt alone (section 4).
- **Page-level citations** are rebuilt from retrieved chunks, so the model can never invent a source or page.
- **Incremental, resumable indexing.** Chunk ids are content hashes: unchanged chunks are skipped, edited or deleted files are cleaned up, progress is saved per batch. This was necessary to survive free-tier rate limits.
- **Gemini client over plain REST** (`requests`): no SDK version conflicts with LangChain, full control over retry, fallback models and daily-quota detection.

## 2. Prompt engineering
- **System prompt per domain** defines role, boundaries (non-diagnostic, no personal dosing, no deceptive marketing), tone and hedging rules.
- **Few-shot examples per domain** show the exact JSON format, a grounded answer, and (medical) an out-of-context refusal.
- **Task presets** (for example `summarize_case`, `interactions`, `campaign_ideas`) add a focused instruction to the same pipeline.
- Hard rules injected for every domain: answer only from CONTEXT, never invent numbers or sources, lower self-confidence when the context does not answer.

## 3. RAG implementation
- **Knowledge base:** public documents per domain (PDF guidelines, drug labels, open textbooks) plus small synthetic samples. PDFs are parsed page by page; CSV/TSV rows are grouped into text blocks.
- **Chunking:** 900 characters, 150 overlap, splitting on headings and paragraphs first. Page/row metadata is copied to every chunk.
- **Retrieval:** embed the question, cosine search in Chroma, top-4 chunks with relevance = 1 - distance.
- **Threshold:** `MIN_RELEVANCE = 0.58`, chosen from the gap between answerable questions (min 0.645) and unrelated questions (max about 0.54) on the sample set. It must be re-calibrated after the real documents are indexed.

## 4. Guardrails and hallucination handling
| Layer | Mechanism | Example |
|---|---|---|
| 1. Input | Regex patterns per domain plus global prompt-injection patterns | "Diagnose me", "how much should I take", "fake reviews" are refused |
| 2. Emergency | Emergency patterns (medical, pharma) | "chest pain" returns an emergency message, not an answer |
| 3. Grounding gate | Best relevance below threshold: no LLM call | "Who won the World Cup?" returns "not enough information", saving quota too |
| 4. Prompt rules | Context-only answering, no invented facts | Model must say when the context lacks the answer |
| 5. Second gate | Model self-confidence below 0.3 becomes `no_context` | Near-domain questions about documents that are not indexed |
| 6. Output validation | JSON schema check, citations restricted to retrieved files | Fake sources are dropped |
| 7. Transparency | Confidence label (High/Medium/Low), citations with pages, disclaimer on every answer | |

**Confidence score** = 60% retrieval strength (mean of top-3 relevance, rescaled 0.50 to 0.80) + 40% model self-reported confidence. High at 0.75 and above, Medium at 0.50 and above, otherwise Low. It is a heuristic signal, not a calibrated probability.

## 5. Evaluation approach
- `evaluation/golden_set.json` holds questions per domain in three types: **answerable** (expected keywords, must be grounded with citations), **blocked** (guardrail must trigger, including prompt injection), **out_of_scope** (must not be answered).
- `evaluation/run_eval.py` reports pass/fail per question and an overall score; `--judge` adds an LLM-as-judge faithfulness score (1 to 5) against the retrieved context.
- **Result on the bundled sample set (smoke test):** Medical 12/12, Pharma 10/10, Marketing 10/10. The evaluation also caught a real bug: a regex typo let a drug-synthesis request through the input guardrail; it was fixed and re-verified.
- **Limitation:** the sample questions were written from the sample documents, so these scores are optimistic. The real evaluation uses 20-30 questions per domain written from the real PDFs, including paraphrased, multi-chunk and adversarial questions. Next step: add RAGAS-style metrics (faithfulness, context precision/recall).

## 6. Trade-offs
| Decision | Gain | Cost |
|---|---|---|
| Gemini free tier (task suggests GPT / Azure OpenAI) | Zero cost, no card | Free quotas are small and models change; model access is isolated in `llm.py`, so moving to GPT or Azure OpenAI means rewriting that one file |
| Regex input guardrails | Fast, free, predictable | Can miss paraphrases or over-block; production needs a classifier or moderation API in front |
| Fixed relevance threshold | Simple, explainable | Needs re-tuning per embedding model and corpus |
| 900-character chunks | Precise retrieval | Can split a thought; overlap mitigates |
| Self-reported confidence | Free signal | Not calibrated |
| Chroma local store | No setup | Single machine, no multi-tenancy |
| Preview/latest model aliases | Avoid shutdowns | Behavior can drift; the evaluation suite is the safety net |

## 7. Scaling approach
- **Retrieval:** move to Azure AI Search or pgvector with hybrid (BM25 + vector) search and a reranker; metadata filters per tenant.
- **Cost/latency:** semantic caching, streaming responses, async calls, batch embedding jobs.
- **Safety:** replace/augment regex with a moderation model; human review queue for low-confidence answers.
- **Multi-tenant:** one index namespace per tenant and domain; domain configs stored in a database instead of YAML.
- **Observability:** log query, retrieved chunks, scores, status, latency; run the golden set in CI on every prompt or model change.
- **Roadmap:** LLM function-calling for tools, a web-search tool, summarized long-term user memory.

## 8. Bonus features
- **Memory-based personalization** (`dagent/memory.py`): a per-user profile (role, answer detail, answer language, last topics) is saved in `user_profiles/<name>.json` and injected into the prompt. It changes tone, depth and language only; guardrails never depend on it. Session memory (last 4 turns) is kept separately in the chat.
- **Tool use: calculator** (`dagent/tools.py`): LLMs are unreliable at arithmetic, so numbers are computed in Python. Safe expression evaluator (AST, no names or calls) plus named formulas per domain (Medical: BMI, MAP; Pharma: percent change, relative risk reduction, NNT; Marketing: ROI, CAC, CPA, conversion rate, ROAS). A small router prompt picks the tool only for calculation-looking questions (one extra LLM call, otherwise zero). The result is passed to the model as a TOOL RESULT it must quote, never recompute. No dosing formulas are offered on purpose.

## 9. Known limitations
Scanned PDFs need OCR; tables in PDFs may extract poorly; large CSVs are not suited to semantic search (summarize them or use a calculator tool); the free API tier limits throughput; sample documents are synthetic.
