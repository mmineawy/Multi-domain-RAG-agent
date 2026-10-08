import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOMAINS_DIR = ROOT / "domains"
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / "chroma_db"

TOP_K = int(os.getenv("TOP_K", "4"))                      # chunks retrieved per question
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.58"))  # grounding gate (calibrated for Gemini embeddings)
