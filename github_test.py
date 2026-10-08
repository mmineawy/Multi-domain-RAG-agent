"""Diagnose GitHub Models. Sends 1-2 tiny requests per endpoint and prints RAW responses.
Run:  python github_test.py        (your token is never printed)"""
import os
import requests
from dagent import config  # loads .env

TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
assert TOKEN, "GITHUB_TOKEN is empty in .env"
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
     "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"}


def show(title, r):
    ct = r.headers.get("content-type", "?")
    body = r.text.replace("\n", " ")[:260]
    print(f"{'✅' if r.status_code == 200 else '❌'} {title}\n   HTTP {r.status_code} | {ct}\n   {body}\n")


def run(title, fn):
    try:
        show(title, fn())
    except Exception as e:
        print(f"❌ {title}\n   {type(e).__name__}: {str(e)[:200]}\n")


print("=== A) models.github.ai (current endpoint) ===")
run("catalog (list models)", lambda: requests.get("https://models.github.ai/catalog/models", headers=H, timeout=30))
run("chat gpt-4o-mini", lambda: requests.post("https://models.github.ai/inference/chat/completions", headers=H, timeout=60,
    json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 10}))
run("embedding text-embedding-3-small", lambda: requests.post("https://models.github.ai/inference/embeddings", headers=H, timeout=60,
    json={"model": "openai/text-embedding-3-small", "input": ["hello"]}))

print("=== B) models.inference.ai.azure.com (older endpoint) ===")
run("chat gpt-4o-mini", lambda: requests.post("https://models.inference.ai.azure.com/chat/completions", headers=H, timeout=60,
    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 10}))
run("embedding text-embedding-3-small", lambda: requests.post("https://models.inference.ai.azure.com/embeddings", headers=H, timeout=60,
    json={"model": "text-embedding-3-small", "input": ["hello"]}))
print("Copy this whole output and send it to Claude.")