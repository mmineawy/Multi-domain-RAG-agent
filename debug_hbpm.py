"""Diagnose why the home-BP-monitoring protocol is missing from the answer.
Run from the project root:  python debug_hbpm.py
Uses 1 embedding call. No chat-model calls."""
import re

from pypdf import PdfReader

from dagent import config
from dagent.rag import _store, retrieve

DOMAIN = "medical"
PHRASE = re.compile(r"consecutive|at least 1 minute|twice daily", re.I)

# 1) How many chunks are indexed? (expected: 471)
print("Indexed chunks:", _store(DOMAIN)._collection.count(), "(expected 471)\n")

# 2) Does the protocol sentence exist in the PDF text, and on which page?
for pdf in sorted((config.KB_DIR / DOMAIN).glob("*.pdf")):
    for i, page in enumerate(PdfReader(str(pdf)).pages, 1):
        text = page.extract_text() or ""
        if "home blood pressure" in text.lower() and PHRASE.search(text):
            m = PHRASE.search(text)
            s = max(0, m.start() - 150)
            print(f"[PDF] {pdf.name[:40]} page {i}: ...{text[s:m.end() + 250]!r}\n")

# 3) What does retrieval return for the original question vs a protocol-focused one?
queries = [
    "If ambulatory monitoring is not suitable, how should home blood pressure monitoring be carried out?",
    "home blood pressure monitoring two consecutive measurements at least 1 minute apart twice daily for 4 days",
]
for q in queries:
    print("QUERY:", q)
    for doc, rel in retrieve(DOMAIN, q):
        hit = "<-- PROTOCOL" if PHRASE.search(doc.page_content) else ""
        print(f"  {rel:.2f}  page {doc.metadata.get('page')}  {doc.page_content[:70]!r} {hit}")
    print()