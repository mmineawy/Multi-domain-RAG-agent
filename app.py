"""Streamlit chat UI.  Run:  streamlit run app.py"""
import streamlit as st

from dagent import DomainAgent, list_domains
from dagent.memory import DETAILS, LANGUAGES, ROLES, load_profile, remember_topic, save_profile
from dagent.rag import ingest

st.set_page_config(page_title="Multi-Domain AI Agent", page_icon="🧠", layout="wide")
ICONS = {"medical": "🩺", "pharma": "💊", "marketing": "📣"}
DOMAINS = list_domains()
COLOR = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


@st.cache_resource
def get_agent(domain: str) -> DomainAgent:
    return DomainAgent(domain)


def pick(options, value):
    return options.index(value) if value in options else 0


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🧠 Multi-Domain Agent")
    user = st.text_input("User name (personal memory)", value="guest").strip() or "guest"
    profile = load_profile(user)
    with st.expander("👤 Personalization"):
        role = st.selectbox("Role", ROLES, index=pick(ROLES, profile["role"]), key=f"role_{user}")
        detail = st.selectbox("Answer detail", DETAILS, index=pick(DETAILS, profile["detail"]), key=f"detail_{user}")
        language = st.selectbox("Answer language", LANGUAGES, index=pick(LANGUAGES, profile["language"]),
                                key=f"lang_{user}")
        if st.button("Forget my history"):
            profile["topics"] = []
        if profile["topics"]:
            st.caption("Remembered topics: " + " | ".join(profile["topics"][-3:]))
    profile.update(role=role, detail=detail, language=language)
    save_profile(user, profile)

    domain = st.selectbox("Domain", DOMAINS, index=pick(DOMAINS, "medical"),
                          format_func=lambda d: f"{ICONS.get(d, '')} {d.title()}")
    agent = get_agent(domain)
    st.caption(agent.cfg["description"])
    task_key = st.selectbox("Task", ["(auto)"] + list(agent.tasks.keys()))
    task_key = None if task_key == "(auto)" else task_key
    if task_key:
        st.info(agent.tasks[task_key])
    runner = lambda q, hist: agent.run(q, task=task_key, history=hist, profile=profile)  # noqa: E731
    chat_key, title = f"chat_{domain}", f"{ICONS.get(domain, '')} {agent.cfg['name']}"
    if st.button("🔄 Update knowledge index"):
        with st.spinner("Indexing (only new/changed chunks)..."):
            st.success(f"{ingest(domain)} chunks in index")
    if st.button("🧹 Clear chat"):
        st.session_state.pop(chat_key, None)
        st.rerun()
    st.divider()
    st.caption("Guardrails: input screening, grounding gate, citations, confidence, disclaimer. "
               "Calculator tool for numeric questions.")


# ---------------- rendering ----------------
def bullets(title, items):
    if items:
        st.markdown(f"**{title}**")
        for x in items:
            st.markdown(f"- {x}")


def refusal(r):
    {"emergency": st.error, "blocked": st.warning}.get(r["status"], st.info)(r["message"])


def render_single(r):
    if r["status"] != "ok":
        refusal(r)
    else:
        a, c = r["answer"], r["confidence"]
        if r.get("tool"):
            st.success(f"🧮 Calculator: {r['tool']['text']}")
        st.markdown(a["summary"])
        bullets("Key points", a["key_points"])
        bullets("Recommendations", a["recommendations"])
        bullets("Risks / cautions", a["risks_or_cautions"])
        st.markdown(f"**Confidence:** {COLOR[c['label']]} {c['label']} ({c['score']})  \n"
                    f"**Sources:** {', '.join(a.get('citations') or a['sources_used'])}")
    if r["sources"]:
        with st.expander("Retrieved passages"):
            for s in r["sources"]:
                st.markdown(f"`{s.get('label', s['source'])}` · relevance **{s['score']}**  \n> {s['snippet']}...")
    st.caption(r["disclaimer"])


render = render_single

# ---------------- chat ----------------
st.header(title)
st.session_state.setdefault(chat_key, [])
for turn in st.session_state[chat_key]:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        render(turn["r"])

if q := st.chat_input("Ask a question (or paste de-identified case notes)..."):
    with st.chat_message("user"):
        st.write(q)
    hist = [{"q": t["q"], "a": t["r"]["answer"]["summary"]} for t in st.session_state[chat_key]
            if t["r"]["status"] == "ok"]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                r = runner(q, hist)
        except Exception as e:  # quota / network problems should not crash the UI
            st.error(f"Model error: {str(e)[:300]}")
            st.stop()
        render(r)
    st.session_state[chat_key].append({"q": q, "r": r})
    save_profile(user, remember_topic(profile, q))
