"""Gemini quota diagnostic: one tiny request per model, no retries.  Run: python quota_test.py"""
import os
import requests
from dagent import config  # loads .env

KEY = os.getenv("GEMINI_API_KEY", "").strip()
assert KEY, "GEMINI_API_KEY is empty in .env"
BASE = "https://generativelanguage.googleapis.com/v1beta"
H = {"x-goog-api-key": KEY}


def msg(r):
    try:
        e = r.json()["error"]
        d = next((x["retryDelay"] for x in e.get("details", []) if "retryDelay" in x), "")
        return f"{e.get('message', '')[:200]} {('| retryDelay=' + d) if d else ''}"
    except Exception:
        return r.text[:200]


names = []
r = requests.get(f"{BASE}/models?pageSize=100", headers=H, timeout=30)
if r.status_code == 200:
    names = [m["name"].replace("models/", "") for m in r.json().get("models", [])]
skip = ("image", "tts", "live", "audio", "native", "robotics", "computer", "thinking")
cands = [os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")]
cands += [n for n in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if n]
cands += [n for n in names if "gemini" in n and "flash" in n and not any(s in n for s in skip)][:5]
for m in dict.fromkeys(cands):
    r = requests.post(f"{BASE}/models/{m}:generateContent", headers=H, timeout=60, json={
        "contents": [{"role": "user", "parts": [{"text": "Reply with OK"}]}], "generationConfig": {"maxOutputTokens": 20}})
    print(f"{'✅' if r.status_code == 200 else '❌'} {m:38s} HTTP {r.status_code} {'' if r.status_code == 200 else msg(r)}")
e = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
r = requests.post(f"{BASE}/models/{e}:embedContent", headers=H, timeout=60, json={"content": {"parts": [{"text": "hello"}]}})
print(f"{'✅' if r.status_code == 200 else '❌'} {e:38s} HTTP {r.status_code} {'' if r.status_code == 200 else msg(r)}")
