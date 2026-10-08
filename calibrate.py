"""Find a good MIN_RELEVANCE using ONLY embedding calls (no LLM, cheap and fast).
Run:  python calibrate.py"""
import json
from pathlib import Path

from dagent import config
from dagent.agent import list_domains
from dagent.rag import retrieve

golden = json.loads((Path(__file__).parent / "evaluation" / "golden_set.json").read_text(encoding="utf-8"))
all_in, all_out = [], []

for d in list_domains():
    print(f"\n=== {d} ===")
    ins, outs = [], []
    for c in golden.get(d, []):
        if c["type"] not in ("answerable", "out_of_scope"):
            continue
        top = max(rel for _, rel in retrieve(d, c["q"]))
        (ins if c["type"] == "answerable" else outs).append(top)
        tag = "IN " if c["type"] == "answerable" else "OUT"
        print(f"  {tag} {top:.3f}  {c['q'][:70]}")
    if ins and outs:
        print(f"  -> answerable: min={min(ins):.3f} | out-of-scope: max={max(outs):.3f}")
    all_in += ins
    all_out += outs

lo, hi = max(all_out), min(all_in)
print("\n" + "=" * 55)
print(f"Lowest answerable score : {hi:.3f}")
print(f"Highest out-of-scope    : {lo:.3f}")
if hi > lo:
    print(f"Clean separation. Suggested MIN_RELEVANCE = {(hi + lo) / 2:.2f}   (currently {config.MIN_RELEVANCE})")
else:
    print(f"Scores OVERLAP (answerable min {hi:.3f} <= out-of-scope max {lo:.3f}).")
    print(f"Pick a value near {(hi + lo) / 2:.2f}; the LLM prompt + confidence label will catch the rest.")
