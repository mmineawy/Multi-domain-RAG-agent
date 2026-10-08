"""Gemini client (chat + embeddings) over Google's REST API using `requests` only.
Includes smart rate-limit handling for the free tier and automatic model fallback."""
import os
import time
from types import SimpleNamespace

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiBusy(RuntimeError):
    """Model overloaded or quota exhausted even after retries."""


def _key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is empty. Create a free key at https://aistudio.google.com/apikey")
    return key


def _retry_delay(resp):
    """Gemini 429 responses often say how long to wait ('retryDelay': '34s')."""
    try:
        for d in resp.json()["error"].get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return None


def _post(url: str, payload: dict, retries: int = 4) -> dict:
    """POST with smart retry: honors retryDelay, fails fast when the DAILY quota is exhausted."""
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers={"x-goog-api-key": _key()}, timeout=90)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(3 * 2 ** attempt)
            continue
        if r.status_code in (429, 500, 503):
            last = f"{r.status_code}: {r.text[:300]}"
            delay = _retry_delay(r) if r.status_code == 429 else None
            if r.status_code == 429 and ("PerDay" in r.text or (delay and delay > 300)):
                hrs = f" Google says retry in about {delay / 3600:.1f} hours." if delay else ""
                raise GeminiBusy(f"DAILY free quota exhausted for this model.{hrs}")
            if attempt < retries - 1:
                wait = min(delay or 3 * 2 ** attempt, 65) + 1
                try:
                    msg = r.json()["error"]["message"][:100]
                except Exception:
                    msg = r.text[:100]
                print(f"[gemini {r.status_code}: {msg} | waiting {wait:.0f}s, retry {attempt + 1}/{retries - 1}]")
                time.sleep(wait)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:400]}")
        return r.json()
    raise GeminiBusy(f"Gemini busy/rate-limited after {retries} tries -> {last}")


class GeminiChat:
    """Chat model: .invoke(messages) -> object with .content. Always answers in JSON mode."""

    def __init__(self, model: str, fallbacks: list[str] | None = None, temperature: float = 0.1):
        self.model, self.fallbacks, self.temperature = model, fallbacks or [], temperature

    def invoke(self, messages):
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        system, contents = [], []
        for m in messages:
            if isinstance(m, SystemMessage):
                system.append(m.content)
            else:
                role = "model" if isinstance(m, AIMessage) else "user"
                contents.append({"role": role, "parts": [{"text": m.content}]})
        payload = {"contents": contents,
                   "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}

        data, last_err = None, None
        for model in [self.model] + self.fallbacks:  # main model first, then fallbacks
            try:
                data = _post(f"{BASE}/models/{model}:generateContent", payload)
                break
            except GeminiBusy as e:
                last_err = e
                print(f"[{model} unavailable -> trying next fallback]")
        if data is None:
            raise last_err
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            text = "{}"  # blocked/empty response -> agent falls back to safe defaults
        return SimpleNamespace(content=text)


class GeminiEmbeddings:
    """Implements embed_documents / embed_query so Chroma can use it (768-dim vectors)."""

    def __init__(self, model: str, dims: int = 768):
        self.model, self.dims = model, dims

    def _embed(self, texts, task):
        out = []
        for i in range(0, len(texts), 50):
            payload = {"requests": [{
                "model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                "taskType": task, "outputDimensionality": self.dims,
            } for t in texts[i:i + 50]]}
            out += [e["values"] for e in _post(f"{BASE}/models/{self.model}:batchEmbedContents", payload)["embeddings"]]
        return out

    def embed_documents(self, texts):
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_llm() -> GeminiChat:
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite-preview,gemini-3-flash-preview")
    return GeminiChat(os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest"),
                      fallbacks=[m.strip() for m in fallbacks.split(",") if m.strip()])


def get_embeddings() -> GeminiEmbeddings:
    return GeminiEmbeddings(os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"))
