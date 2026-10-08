# Knowledge base

Put documents for each domain in its own folder: `knowledge_base/medical/`, `pharma/`, `marketing/`.
Supported: `.pdf` (page numbers kept for citations), `.md`, `.txt`, `.csv`, `.tsv`. Scanned (image-only) PDFs are not supported.
After adding or changing files run `python ingest.py <domain>`: only new or changed chunks are embedded.

## Sources used (public, free)
Add the exact document titles and links here when you finalize the knowledge base, for example:
- Medical: NICE NG136 (Hypertension in adults), NICE NG28 (Type 2 diabetes in adults)
- Pharma: FDA/DailyMed drug labels, FDA drug-interaction guidance
- Marketing: DataReportal Digital 2026 Egypt, OpenStax Principles of Marketing (selected chapters)
