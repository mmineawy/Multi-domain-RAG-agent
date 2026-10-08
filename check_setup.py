"""Health check. Run from the project root:  python check_setup.py
Uses very few API calls; it never builds the index (run `python ingest.py` for that)."""
import importlib.metadata as md
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(True)
        print(f"✅ {name}" + (f" - {detail}" if detail else ""))
        return True
    except Exception as e:
        results.append(False)
        print(f"❌ {name}\n     -> {type(e).__name__}: {str(e)[:300]}")
        return False


def py():
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    return sys.version.split()[0]


def venv():
    assert sys.prefix != sys.base_prefix, "virtual environment is not active"
    return "active"


def key():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    k = os.getenv("GEMINI_API_KEY", "").strip()
    assert k and "paste-your" not in k, "GEMINI_API_KEY missing in .env"
    return f"...{k[-4:]}"


def domains():
    from dagent.agent import list_domains, load_domain
    ds = list_domains()
    [load_domain(d) for d in ds]
    return ", ".join(ds)


def kb():
    from dagent import config
    ok = {".md", ".txt", ".pdf", ".csv", ".tsv"}
    out = []
    for d in sorted(p for p in config.KB_DIR.iterdir() if p.is_dir()):
        n = len([f for f in d.iterdir() if f.suffix.lower() in ok])
        assert n, f"no supported files in knowledge_base/{d.name}"
        out.append(f"{d.name}={n}")
    return ", ".join(out)


def chat():
    from dagent.llm import get_llm
    return get_llm().invoke("Reply with a JSON object: {\"ok\": true}").content.strip()[:40]


def embed():
    from dagent.llm import get_embeddings
    return f"vector size {len(get_embeddings().embed_query('hello'))}"


def index_state():
    from dagent.agent import list_domains
    from dagent.rag import _store
    info = {d: _store(d)._collection.count() for d in list_domains()}
    empty = [d for d, n in info.items() if n == 0]
    if empty:
        raise AssertionError(f"chunks per domain {info} -> run: python ingest.py")
    return f"chunks per domain {info}"


def retrieval_and_agent():
    from dagent import DomainAgent
    from dagent.rag import retrieve
    hits = retrieve("medical", "How is hypertension defined?")
    r = DomainAgent("medical").run("How is hypertension defined?")
    assert r["status"] == "ok", f"status={r['status']}"
    return f"top match {hits[0][0].metadata['source']} ({hits[0][1]:.2f}); confidence {r['confidence']['label']}"


def guardrails():
    from dagent import DomainAgent
    a = DomainAgent("medical")
    assert a.run("Diagnose me, do I have diabetes?")["status"] == "blocked", "diagnosis not blocked"
    assert a.run("I have chest pain right now")["status"] == "emergency", "emergency not detected"


def calculator():
    from dagent.tools import run_formula, safe_eval
    assert run_formula("bmi", {"weight_kg": 82, "height_cm": 175})["value"] == 26.78
    assert safe_eval("2+3*4") == 14
    try:
        safe_eval("__import__('os')")
        raise AssertionError("unsafe expression accepted")
    except ValueError:
        pass
    return "bmi and safe arithmetic OK"


print("=== Environment ===")
check("Python", py)
check("Virtual env", venv)
print("\n=== Packages ===")
pk = all([check(p, lambda p=p: md.version(p)) for p in
          ["langchain-core", "langchain-text-splitters", "langchain-chroma", "chromadb",
           "pyyaml", "python-dotenv", "requests", "pypdf", "streamlit"]])
print("\n=== Configuration ===")
cfg = pk and check("Gemini API key", key) and check("Domain configs", domains)
if cfg:
    check("Knowledge base files", kb)
    check("Calculator tool (no API call)", calculator)
    print("\n=== Gemini connection ===")
    if check("Chat model", chat) and check("Embedding model", embed):
        print("\n=== Pipeline ===")
        if check("Vector index", index_state):
            check("Retrieval + agent answer", retrieval_and_agent)
            check("Guardrails (no API call)", guardrails)


failed = results.count(False)
print("\n" + "=" * 45)
print(f"RESULT: {results.count(True)} passed, {failed} failed")
print("All good." if not failed else "Fix the ❌ items above and run again.")
