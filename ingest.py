"""Build / update the vector index (incremental: only new or changed files are embedded).
Usage:  python ingest.py                  update all domains
        python ingest.py medical          update one domain
        python ingest.py medical --rebuild   wipe that domain and re-embed everything"""
import sys
from dagent import list_domains
from dagent.rag import ingest

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rebuild = "--rebuild" in sys.argv
for d in (args or list_domains()):
    print(f"[{d}]")
    n = ingest(d, rebuild=rebuild)
    print(f"[{d}] done - {n} chunks in index\n")
