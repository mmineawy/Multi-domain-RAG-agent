"""One generic agent. Behaviour is defined by domains/<name>.yaml + knowledge_base/<name>/."""
import json
import re

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config, guardrails, tools
from .llm import get_llm
from .memory import profile_prompt
from .rag import label, retrieve

SCHEMA_INSTRUCTIONS = """
OUTPUT FORMAT - respond with ONE valid JSON object and nothing else:
{
  "summary": "2-4 sentence answer grounded in the context",
  "key_points": ["short bullet", "..."],
  "recommendations": ["actionable item (framed as options/considerations, not orders)", "..."],
  "risks_or_cautions": ["risk, limitation or interaction", "..."],
  "sources_used": ["file names from the context that you actually used"],
  "self_confidence": 0.0-1.0
}
RULES:
- Use ONLY the CONTEXT. If the context does not contain the answer, say so in "summary", leave lists empty and set self_confidence below 0.3.
- Never invent facts, numbers, doses, or citations.
- In "sources_used" list only file names (without page numbers) that appear in the context.
- If a TOOL RESULT is given, quote that exact number; never recompute it. Interpret it only with the CONTEXT.
- If a USER PROFILE is given, adapt depth, tone and language to it, but never relax any safety rule.
"""


def list_domains() -> list[str]:
    return sorted(p.stem for p in config.DOMAINS_DIR.glob("*.yaml"))


def load_domain(name: str) -> dict:
    path = config.DOMAINS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown domain '{name}'. Available: {list_domains()}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _format_context(hits) -> str:
    return "\n\n".join(f"[{i}] source: {label(d.metadata)}\n{d.page_content.strip()}"
                       for i, (d, _) in enumerate(hits, 1))


class DomainAgent:
    def __init__(self, domain: str):
        self.domain = domain
        self.cfg = load_domain(domain)
        self.llm = get_llm()

    @property
    def tasks(self) -> dict:
        return self.cfg.get("tasks", {})

    def _messages(self, query, context, task_key, history, tool=None, profile=None):
        msgs = [SystemMessage(content=self.cfg["system_prompt"].strip() + "\n" + SCHEMA_INSTRUCTIONS)]
        for ex in self.cfg.get("few_shot", []):  # few-shot examples
            msgs.append(HumanMessage(content=ex["user"].strip()))
            msgs.append(AIMessage(content=ex["assistant"].strip()))
        for turn in (history or [])[-4:]:  # session memory
            msgs.append(HumanMessage(content=turn["q"]))
            msgs.append(AIMessage(content=turn["a"]))
        extras = ""
        if tool:
            extras += f"TOOL RESULT (computed exactly by the calculator): {tool['text']}\n\n"
        if task_key in self.tasks:
            extras += f"TASK: {self.tasks[task_key]}\n\n"
        if profile:  # persistent user memory
            extras += profile_prompt(profile) + "\n\n"
        msgs.append(HumanMessage(content=f"CONTEXT:\n{context}\n\n{extras}QUESTION: {query}"))
        return msgs

    @staticmethod
    def _calc_answer(tool: dict) -> dict:
        return {"summary": tool["text"], "key_points": [], "recommendations": [],
                "risks_or_cautions": ["Computed exactly by the built-in calculator. Interpretation needs the "
                                      "knowledge base or a qualified professional."],
                "sources_used": [], "citations": ["Built-in calculator"], "self_confidence": 1.0}

    def run(self, query: str, task: str | None = None, history: list | None = None,
            profile: dict | None = None) -> dict:
        base = {"domain": self.domain, "query": query, "answer": None, "sources": [], "confidence": None,
                "contexts": [], "message": "", "tool": None, "disclaimer": self.cfg["disclaimer"]}

        # 1) Input guardrail
        status, msg = guardrails.check_input(query, self.cfg)
        if status != "ok":
            return {**base, "status": status, "message": msg}

        # 2) Calculator tool (only for calculation-looking questions)
        tool = tools.plan_and_run(self.llm, self.domain, query)
        base["tool"] = tool
        calc = {"score": 1.0, "label": "High", "retrieval": 0.0, "self": 1.0}

        # 3) Retrieval
        hits = retrieve(self.domain, query)
        sources = [{"source": d.metadata["source"], "label": label(d.metadata), "page": d.metadata.get("page"),
                    "chunk_id": d.metadata.get("chunk_id"), "score": round(r, 2),
                    "snippet": d.page_content[:220].replace("\n", " ")} for d, r in hits]
        base["sources"] = sources
        base["contexts"] = [d.page_content for d, _ in hits]

        # 4) Grounding gate: weak retrieval -> refuse to guess (a pure calculation still succeeds)
        if not hits or max(r for _, r in hits) < config.MIN_RELEVANCE:
            if tool:
                return {**base, "status": "ok", "answer": self._calc_answer(tool), "confidence": calc}
            return {**base, "status": "no_context",
                    "message": "I don't have enough information in my knowledge base to answer this reliably.",
                    "confidence": {"score": 0.0, "label": "Low", "retrieval": 0.0, "self": 0.0}}

        # 5) LLM reasoning
        raw = self.llm.invoke(self._messages(query, _format_context(hits), task, history, tool, profile)).content
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {}
        answer = guardrails.validate_output(parsed)

        # 6) Output guardrail: keep only citations that really were retrieved
        valid = {s["source"] for s in sources}
        norm = [re.sub(r"\s*\((page|rows)[^)]*\)\s*$", "", s).strip() for s in answer["sources_used"]]
        answer["sources_used"] = list(dict.fromkeys(s for s in norm if s in valid)) or sorted(valid)[:2]
        answer["citations"] = sorted({s["label"] for s in sources if s["source"] in answer["sources_used"]})
        conf = guardrails.compute_confidence([r for _, r in hits], answer["self_confidence"])

        # 7) Second gate: the model itself says the context does not answer the question
        try:
            low_self = float(answer["self_confidence"]) < 0.3
        except (TypeError, ValueError):
            low_self = False
        if low_self:
            if tool:
                return {**base, "status": "ok", "answer": self._calc_answer(tool), "confidence": calc}
            return {**base, "status": "no_context", "confidence": conf,
                    "message": answer["summary"] or "The knowledge base does not contain a reliable answer."}

        if tool:
            answer["citations"] = ["Built-in calculator"] + answer["citations"]
        return {**base, "status": "ok", "answer": answer, "confidence": conf}
