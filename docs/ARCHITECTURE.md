# Architecture

## End-to-end flow
```mermaid
flowchart TD
    U[User - Streamlit chat or CLI] --> IG{Input guardrail}
    CFG[(domains/domain.yaml<br/>system prompt, few-shot,<br/>patterns, disclaimer)] -.-> IG
    IG -- blocked --> R1[Polite refusal]
    IG -- emergency --> R2[Emergency message]
    IG -- ok --> EQ[Embed question<br/>Gemini embedding-001]
    EQ --> VS[(ChromaDB<br/>one collection per domain)]
    DOCS[(knowledge_base/domain/<br/>PDF, MD, TXT, CSV)] --> ING[Ingest: parse, chunk 900/150,<br/>embed, store - incremental] --> VS
    VS --> TOPK[Top-k chunks + relevance scores]
    TOPK --> GG{Grounding gate<br/>best score >= MIN_RELEVANCE?}
    GG -- no --> R3["Not enough information"]
    GG -- yes --> PR[Prompt: system + few-shot +<br/>session memory + context + task]
    CFG -.-> PR
    PR --> LLM[Gemini chat model<br/>JSON mode, fallback models]
    LLM --> V[Output validation<br/>schema + real citations only]
    V --> G2{Model self-confidence<br/>below 0.3?}
    G2 -- yes --> R3
    G2 -- no --> CS[Confidence score<br/>60% retrieval + 40% self-reported]
    CS --> OUT[Structured answer + citations<br/>file, page + confidence + disclaimer]
    OUT --> U
```

## Same agent, three domains
```mermaid
flowchart LR
    A[Generic agent<br/>dagent/agent.py] --> M[medical.yaml + knowledge_base/medical]
    A --> P[pharma.yaml + knowledge_base/pharma]
    A --> K[marketing.yaml + knowledge_base/marketing]
```
Switching domain changes only: system prompt, few-shot examples, task list, blocked/emergency patterns, disclaimer, and which vector collection is searched.

## Components
| Component | Choice | Role |
|---|---|---|
| LLM | Gemini (`gemini-flash-lite-latest`, automatic fallbacks) | Reasoning, structured JSON answer |
| Embeddings | `gemini-embedding-001`, 768 dims | Semantic search |
| Vector DB | ChromaDB (cosine, persistent, one collection per domain) | Retrieval |
| Parsing / chunking | pypdf, csv, `RecursiveCharacterTextSplitter` (900 chars, 150 overlap) | Page-aware chunks |
| Orchestration | LangChain core, text-splitters, chroma | Messages, splitting, vector store |
| Config | YAML per domain, `.env` for keys and thresholds | Domain adaptation |
| UI | Streamlit | Chat, domain/task switcher, sources |

## Structured output
```json
{"summary": "...", "key_points": [], "recommendations": [], "risks_or_cautions": [],
 "sources_used": ["file.pdf"], "self_confidence": 0.0}
```
The agent adds `citations` (for example `file.pdf (page 12)`) built from the chunks actually retrieved, plus `confidence {label, score}` and the domain disclaimer.

## Bonus: calculator tool and personalization inside the pipeline
```mermaid
flowchart LR
    Q[Question] --> C{Looks like a calculation?<br/>digits + keywords}
    C -- no --> N[Normal RAG pipeline]
    C -- yes --> RT[Router prompt picks tool + numbers]
    RT --> T[tools.py: formula or safe arithmetic]
    T --> N
    N --> LL[LLM receives CONTEXT + TOOL RESULT + USER PROFILE]
    P[(user_profiles/name.json<br/>role, detail, language, topics)] -.-> LL
```
If retrieval finds nothing relevant but the calculator succeeded, the exact calculation is still returned (citation: Built-in calculator).
