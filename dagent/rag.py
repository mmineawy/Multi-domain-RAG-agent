"""RAG layer: load .md/.txt/.pdf/.csv/.tsv per domain, chunk, embed into Chroma, retrieve with relevance scores."""
import csv
import hashlib
import re
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .llm import GeminiBusy, get_embeddings

SUPPORTED = {".md", ".txt", ".pdf", ".csv", ".tsv"}


def _store(domain: str) -> Chroma:
    return Chroma(
        collection_name=f"kb_{domain}_gemini",
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR / domain),
        collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )


def label(meta: dict) -> str:
    """Human-readable citation, e.g. 'NICE_NG136.pdf (page 12)'."""
    if meta.get("page"):
        return f"{meta['source']} (page {meta['page']})"
    if meta.get("rows"):
        return f"{meta['source']} (rows {meta['rows']})"
    return meta["source"]


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)   # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_text(path: Path):
    return [Document(page_content=path.read_text(encoding="utf-8", errors="ignore"),
                     metadata={"source": path.name})], "1 text file"


def _load_pdf(path: Path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if len(text) >= 40:  # skip blank / image-only pages
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
    return docs, f"{len(reader.pages)} pages ({len(docs)} with text)"


def _load_csv(path: Path, rows_per_doc: int = 10):
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    try:
        delim = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    rows = list(csv.DictReader(raw.splitlines(), delimiter=delim))
    docs = []
    for start in range(0, len(rows), rows_per_doc):
        grp = rows[start:start + rows_per_doc]
        text = "\n".join("; ".join(f"{k}: {v}" for k, v in r.items() if v not in (None, "")) for r in grp)
        docs.append(Document(page_content=text,
                             metadata={"source": path.name, "rows": f"{start + 1}-{start + len(grp)}"}))
    return docs, f"{len(rows)} rows"


LOADERS = {".md": _load_text, ".txt": _load_text, ".pdf": _load_pdf, ".csv": _load_csv, ".tsv": _load_csv}


def load_documents(domain: str) -> list[Document]:
    kb_path = config.KB_DIR / domain
    files = sorted(p for p in kb_path.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No supported files {sorted(SUPPORTED)} in {kb_path}")
    docs = []
    for p in files:
        try:
            d, info = LOADERS[p.suffix.lower()](p)
        except Exception as e:  # one bad file must not break the whole index
            print(f"  [skip] {p.name}: {type(e).__name__}: {e}")
            continue
        if not d:
            print(f"  [warn] {p.name}: no readable text (scanned PDF? needs OCR)")
            continue
        for x in d:
            x.metadata["domain"] = domain
        print(f"  [ok]   {p.name}: {info}")
        docs += d
    return docs


def _chunk_id(c: Document) -> str:
    """Stable id from source + position + text: same content -> same id, so re-runs skip it."""
    key = f"{c.metadata['source']}|{c.metadata.get('page', '')}|{c.metadata.get('rows', '')}|{c.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def ingest(domain: str, rebuild: bool = False, batch_size: int = 32) -> int:
    """Incremental + resumable indexing.
    - only NEW or CHANGED chunks are embedded (saves free-tier quota)
    - chunks of deleted/edited files are removed
    - progress is saved after every batch, so an interruption or 429 never loses finished work
    - rebuild=True wipes the domain index first"""
    docs = load_documents(domain)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)  # metadata (source/page/rows) is copied to every chunk
    wanted: dict[str, Document] = {}
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        wanted.setdefault(_chunk_id(c), c)

    store = _store(domain)
    if rebuild:
        store.delete_collection()
        store = _store(domain)
    existing = set(store._collection.get(include=[])["ids"])

    stale = existing - set(wanted)
    if stale:
        store._collection.delete(ids=list(stale))
    todo = [(i, c) for i, c in wanted.items() if i not in existing]
    print(f"  chunks: {len(wanted)} total | {len(wanted) - len(todo)} already indexed | "
          f"{len(todo)} to embed | {len(stale)} stale removed")

    for b in range(0, len(todo), batch_size):
        batch = todo[b:b + batch_size]
        for attempt in range(15):
            try:
                store.add_documents([c for _, c in batch], ids=[i for i, _ in batch])
                break
            except Exception as e:
                if "exhausted" in str(e):  # daily quota gone: waiting minutes won't help
                    done = b
                    raise SystemExit(f"\nDaily free embedding quota is exhausted ({done}/{len(todo)} chunks saved).\n"
                                     f"{e}\nRun the same command again later; it resumes from chunk {done}.")
                if not (isinstance(e, GeminiBusy) or "429" in str(e) or "rate" in str(e).lower()):
                    raise
                print(f"  [rate limited - waiting 60s, then retrying the same batch ({attempt + 1}/15)]")
                time.sleep(60)
        else:
            raise RuntimeError("Still rate-limited after 15 waits. Run ingest again later; it will resume.")
        print(f"  embedded {min(b + batch_size, len(todo))}/{len(todo)}")
    return len(wanted)


def retrieve(domain: str, query: str, k: int | None = None):
    """Return list of (Document, relevance 0..1). Auto-ingests on first use."""
    k = k or config.TOP_K
    store = _store(domain)
    if store._collection.count() == 0:
        ingest(domain, rebuild=False)
        store = _store(domain)
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, max(0.0, min(1.0, 1.0 - float(dist)))) for doc, dist in results]
