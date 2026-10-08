"""Calculator tool. LLMs are unreliable at arithmetic, so numbers are computed here, exactly.
Pure Python, no API. A router prompt picks the tool only for calculation-looking questions."""
import ast
import json
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """Evaluate plain arithmetic only (no names, calls or attributes)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    clean = expr.replace(",", "").replace("^", "**").replace("%", "/100").replace("x", "*")
    return ev(ast.parse(clean.strip(), mode="eval"))


# name: (args, function, unit, description)
FORMULAS = {
    "bmi": (["weight_kg", "height_cm"], lambda w, h: w / ((h / 100) ** 2), "kg/m2", "Body mass index"),
    "map": (["systolic", "diastolic"], lambda s, d: (s + 2 * d) / 3, "mmHg", "Mean arterial pressure"),
    "percent_change": (["old", "new"], lambda o, n: (n - o) / o * 100, "%", "Percent change from old to new"),
    "relative_risk_reduction": (["control_rate_pct", "treatment_rate_pct"], lambda c, t: (c - t) / c * 100, "%",
                                "Relative risk reduction from event rates in percent"),
    "nnt": (["control_rate_pct", "treatment_rate_pct"], lambda c, t: 100 / (c - t), "patients",
            "Number needed to treat from event rates in percent"),
    "roi": (["revenue", "cost"], lambda r, c: (r - c) / c * 100, "%", "Return on investment"),
    "cac": (["spend", "new_customers"], lambda s, n: s / n, "per customer", "Customer acquisition cost"),
    "cpa": (["spend", "conversions"], lambda s, c: s / c, "per conversion", "Cost per acquisition"),
    "conversion_rate": (["conversions", "visitors"], lambda c, v: c / v * 100, "%", "Conversion rate"),
    "roas": (["revenue", "ad_spend"], lambda r, a: r / a, "x", "Return on ad spend"),
}
DOMAIN_TOOLS = {
    "medical": ["bmi", "map", "percent_change"],
    "pharma": ["percent_change", "relative_risk_reduction", "nnt"],
    "marketing": ["roi", "cac", "cpa", "conversion_rate", "roas", "percent_change"],
}


def run_formula(name: str, args: dict) -> dict:
    arg_names, fn, unit, desc = FORMULAS[name]
    vals = [float(args[a]) for a in arg_names]  # KeyError/ValueError if an input is missing
    value = round(fn(*vals), 2)
    inputs = ", ".join(f"{a}={args[a]}" for a in arg_names)
    return {"tool": name, "value": value, "unit": unit, "text": f"{desc} = {value} {unit} [{inputs}]"}


def run_expression(expr: str) -> dict:
    value = round(float(safe_eval(expr)), 4)
    return {"tool": "expression", "value": value, "unit": "", "text": f"{expr.strip()} = {value}"}


_CALC_WORDS = r"\b(calculate|compute|calc|bmi|map|roi|cac|cpa|roas|nnt|conversion rate|percent|percentage|how much is)\b"


def looks_like_calc(query: str) -> bool:
    q = query.lower()
    return bool(re.search(r"\d", q)) and bool(re.search(_CALC_WORDS, q) or re.search(r"\d\s*[+\-*/x]\s*\d", q))


def plan_and_run(llm, domain: str, query: str):
    """Return a tool result dict or None. Costs one extra LLM call, only for calculation-looking questions."""
    if not looks_like_calc(query):
        return None
    specs = "\n".join(f"- {n}({', '.join(FORMULAS[n][0])}): {FORMULAS[n][3]}" for n in DOMAIN_TOOLS.get(domain, []))
    prompt = ("You are a tool router. Decide whether the question asks for a calculation.\n"
              f"Available tools:\n{specs}\n- expression: plain arithmetic (numbers, + - * / ** parentheses, %)\n"
              'Respond with ONE JSON object: {"tool": "<tool name>" | "expression" | "none", '
              '"args": {"<arg>": number}, "expression": "<arithmetic>"}\n'
              'Use "none" if no calculation is requested or numbers are missing. Never guess missing numbers.\n\n'
              f"QUESTION: {query}")
    try:
        plan = json.loads(llm.invoke(prompt).content)
        tool = plan.get("tool")
        if tool in DOMAIN_TOOLS.get(domain, []):
            return run_formula(tool, plan.get("args") or {})
        if tool == "expression" and plan.get("expression"):
            return run_expression(plan["expression"])
    except Exception:
        return None  # any failure: continue as a normal RAG question
    return None
