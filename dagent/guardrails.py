"""Guardrails: input screening, retrieval-grounding gate, confidence scoring, disclaimers."""
import re

RETRIEVAL_FLOOR, RETRIEVAL_CEIL = 0.50, 0.80

# Applied to every domain
GLOBAL_BLOCKED = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"reveal (your )?(system )?prompt",
    r"you are now (?!a clinical|a pharma|a marketing)",
]


def check_input(query: str, cfg: dict):
    """Return (status, message). status: ok | blocked | emergency."""
    q = query.lower()
    for pat in cfg.get("emergency_patterns", []):
        if re.search(pat, q):
            return "emergency", cfg["emergency_message"]
    for pat in GLOBAL_BLOCKED + cfg.get("blocked_patterns", []):
        if re.search(pat, q):
            return "blocked", cfg["refusal_message"]
    return "ok", ""


def compute_confidence(relevances: list[float], self_conf: float | None):
    """Blend retrieval strength (60%) with the model's self-reported confidence (40%)."""
    top = sorted(relevances, reverse=True)[:3]
    retrieval = sum(top) / len(top) if top else 0.0
    # Calibrated for Gemini embeddings: weak-but-passing matches ~0.55-0.65, strong ones 0.75+.
    # Map 0.50 -> 0 and 0.80 -> 1 so the retrieval part of the score actually varies.
    retrieval_scaled = max(0.0, min(1.0, (retrieval - RETRIEVAL_FLOOR) / (RETRIEVAL_CEIL - RETRIEVAL_FLOOR)))
    try:
        s = float(self_conf)
        s = max(0.0, min(1.0, s))
    except (TypeError, ValueError):
        s = 0.5
    score = round(0.6 * retrieval_scaled + 0.4 * s, 2)
    label = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    return {"score": score, "label": label, "retrieval": round(retrieval, 2), "self": round(s, 2)}


def validate_output(parsed: dict):
    """Make sure the model returned the expected structure; fill safe defaults."""
    parsed = parsed if isinstance(parsed, dict) else {}
    return {
        "summary": str(parsed.get("summary", "")).strip(),
        "key_points": [str(x) for x in parsed.get("key_points", []) if x],
        "recommendations": [str(x) for x in parsed.get("recommendations", []) if x],
        "risks_or_cautions": [str(x) for x in parsed.get("risks_or_cautions", []) if x],
        "sources_used": [str(x) for x in parsed.get("sources_used", []) if x],
        "self_confidence": parsed.get("self_confidence", 0.5),
    }
