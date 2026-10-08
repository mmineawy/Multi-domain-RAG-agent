# Multi-Domain RAG Agent (Medical · Pharma · Marketing)

One generic AI agent that answers from a private knowledge base (RAG) and adapts to a new industry by swapping **one YAML file and one document folder**. No domain logic lives in the Python code.

| Domain | Use case | Tasks |
|---|---|---|
| 🩺 **Medical** (main) | Clinical decision support, non-diagnostic | Summarize case notes, suggest guideline considerations, list red flags |
| 💊 Pharma | Drug research assistant | Summarize papers, highlight risks and interactions, safety profile |
| 📣 Marketing | Campaign strategist | Campaign ideas, customer-segment analysis, channel plan |

**Stack:** Python 3.10+ · Gemini API (chat + embeddings) · ChromaDB · LangChain components (core, text-splitters, chroma) · Streamlit · pypdf

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env  -> add your free GEMINI_API_KEY
python check_setup.py             # health check
python ingest.py                  # build/update the vector index (incremental, resumable)
streamlit run app.py              # chat UI; switch domain/task in the sidebar
```
Free API key (no card): https://aistudio.google.com/apikey

## Commands
| Command | Purpose |
|---|---|
| `python check_setup.py` | Verify environment, key, models, index, guardrails |
| `python ingest.py [domain] [--rebuild]` | Index documents; only new/changed chunks are embedded |
| `python cli.py --domain medical` | Terminal chat (`/domain pharma`, `/task red_flags`, `/quit`) |
| `python -m evaluation.run_eval --domain medical [--judge]` | Run the golden set, write `evaluation/results_<domain>.md` |
| `python calibrate.py` | Compute a data-driven `MIN_RELEVANCE` from the golden set |
| `python -m evaluation.question_forge --domain medical --n 10` | Draft test questions from your indexed documents (review by hand), then `run_eval --file forged_medical.json` |
| `python quota_test.py` | Diagnose Gemini quota / model availability |

## Project structure
```
app.py                    Streamlit chat UI (domain + task switcher, citations, confidence)
cli.py                    Terminal chat
ingest.py                 Incremental indexing
check_setup.py            Health check
calibrate.py              Relevance-threshold calibration
quota_test.py             Gemini diagnostics
dagent/
  agent.py                The single generic agent (pipeline)
  rag.py                  Loaders (md/txt/pdf/csv/tsv), chunking, Chroma, retrieval
  guardrails.py           Input screening, confidence scoring, output validation
  tools.py                Calculator tool (safe arithmetic + named formulas per domain)
  memory.py               Persistent user profile (personalization)
  llm.py                  Gemini chat + embeddings client (retry, fallback models)
  config.py               Settings from .env
domains/*.yaml            System prompt, few-shot, tasks, guardrail patterns, disclaimer
knowledge_base/<domain>/  Documents per domain (PDF, MD, TXT, CSV)
knowledge_base_samples/   Synthetic sample documents
evaluation/               Golden set, evaluation runner, Question Forge (test-set drafting)
user_profiles/            Local per-user memory (created at runtime, git-ignored)
docs/                     Architecture, design document, Arabic project guide
```

## How the agent works
`Input guardrail → retrieve top-k chunks → grounding gate → LLM (system prompt + few-shot + context, JSON output) → validate and keep only real citations → confidence score → disclaimer`.
Details and diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Decisions, trade-offs and scaling: [docs/DESIGN.md](docs/DESIGN.md).

## Add a new domain (e.g. Finance)
1. Copy `domains/medical.yaml` to `domains/finance.yaml` and edit prompt, tasks, patterns, disclaimer.
2. Put documents in `knowledge_base/finance/`.
3. `python ingest.py finance`. It appears in the UI automatically.

## Assignment coverage
| Requirement | Where |
|---|---|
| LLM reasoning | `dagent/llm.py`, `dagent/agent.py` (Gemini) |
| RAG (knowledge base + retrieval logic) | `dagent/rag.py`, `knowledge_base/` |
| Prompt engineering: system prompt per domain + few-shot | `domains/*.yaml` |
| Guardrails: safety, accuracy, disclaimer, confidence | `dagent/guardrails.py`, `domains/*.yaml` |
| Structured output | JSON schema in `dagent/agent.py` |
| Domain adaptation (same agent, 3 domains) | `domains/` + sidebar switcher |
| Architecture diagram | `docs/ARCHITECTURE.md` |
| Evaluation approach, hallucination handling | `evaluation/`, `docs/DESIGN.md` |
| Design decisions, trade-offs, scaling | `docs/DESIGN.md` |
| Bonus: chat UI | `app.py` (Streamlit) |
| Bonus: memory-based personalization | `dagent/memory.py` (role, detail level, language, recent topics, persisted per user) + session memory |
| Bonus: use of tools | `dagent/tools.py` (calculator: BMI, MAP, ROI, CAC, NNT, safe arithmetic) |

## Data and safety notes
- Use public documents only (guidelines, drug labels, open textbooks). Never index real patient or customer data.
- Do not commit PDFs, `.env` or `chroma_db/`. List document sources in this README instead.
- This is decision support, not medical or financial advice. Every answer carries a disclaimer and a confidence label.
