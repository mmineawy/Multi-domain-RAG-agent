"""Question Forge: drafts a test set from YOUR indexed documents (one question + expected keywords per passage).
Usage:  python -m evaluation.question_forge --domain medical --n 10 [--seed 7]
Output: evaluation/forged_<domain>.json  (answerable questions + the blocked/out-of-scope cases from golden_set.json)
Review it by hand, then:  python -m evaluation.run_eval --domain medical --file forged_medical.json"""
import argparse
import json
import random
from pathlib import Path

from dagent.llm import get_llm
from dagent.rag import _store, label

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--n", type=int, default=10, help="number of answerable questions")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = _store(args.domain)._collection.get(include=["documents", "metadatas"])
    items = [(d, m) for d, m in zip(data["documents"], data["metadatas"]) if len(d) > 400]
    if not items:
        raise SystemExit(f"No indexed passages for '{args.domain}'. Run: python ingest.py {args.domain}")
    random.Random(args.seed).shuffle(items)
    picks = items[:args.n]

    llm, forged = get_llm(), []
    for i in range(0, len(picks), 5):  # 5 passages per LLM call saves quota
        batch = picks[i:i + 5]
        passages = "\n\n".join(f"[{j}] {d[:1200]}" for j, (d, _) in enumerate(batch))
        prompt = ("For each passage write ONE question that can be answered ONLY from that passage, phrased like a "
                  "real user (never mention 'the passage'), plus 2-3 short lowercase keywords or numbers that a "
                  'correct answer must contain. Respond with JSON: {"items": [{"id": 0, "question": "...", '
                  f'"keywords": ["..."]}}, ...]}}\n\nPASSAGES:\n{passages}')
        for it in json.loads(llm.invoke(prompt).content).get("items", []):
            try:
                meta = batch[int(it["id"])][1]
                forged.append({"q": it["question"], "type": "answerable", "keywords": it["keywords"],
                               "source": label(meta)})
            except (KeyError, IndexError, ValueError, TypeError):
                continue

    golden = json.loads((HERE / "golden_set.json").read_text(encoding="utf-8")).get(args.domain, [])
    forged += [c for c in golden if c["type"] != "answerable"]  # keep hand-written safety/out-of-scope cases
    out = HERE / f"forged_{args.domain}.json"
    out.write_text(json.dumps({args.domain: forged}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.name}: {sum(c['type'] == 'answerable' for c in forged)} answerable + "
          f"{sum(c['type'] != 'answerable' for c in forged)} guardrail cases. Review it before use.")


if __name__ == "__main__":
    main()
