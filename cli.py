"""Terminal chat.  Usage: python cli.py --domain medical
Commands: /domain <name>   /task <key>   /quit"""
import argparse

from dagent import DomainAgent, list_domains

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="medical", choices=list_domains())
args = ap.parse_args()

agent, history, task = DomainAgent(args.domain), [], None
print("Commands: /domain <name>, /task <key>, /quit | Tasks:", ", ".join(agent.tasks))


def show_list(title, items):
    if items:
        print(f"\n{title}:")
        for x in items:
            print(" -", x)


while True:
    q = input("\nYou> ").strip()
    if not q:
        continue
    if q == "/quit":
        break
    if q.startswith("/domain "):
        agent, history = DomainAgent(q.split()[1]), []
        print("Switched to", agent.domain)
        continue
    if q.startswith("/task "):
        task = q.split()[1]
        print("Task =", task)
        continue
    r = agent.run(q, task=task, history=history)
    if r["status"] != "ok":
        print(f"\n[{r['status'].upper()}] {r['message']}")
        continue
    a = r["answer"]
    if r.get("tool"):
        print("\nCALCULATOR:", r["tool"]["text"])
    print("\nSUMMARY:", a["summary"])
    show_list("KEY POINTS", a["key_points"])
    show_list("RECOMMENDATIONS", a["recommendations"])
    print("\nSOURCES:", ", ".join(a.get("citations") or a["sources_used"]))
    history.append({"q": q, "a": a["summary"]})
    show_list("RISKS / CAUTIONS", a["risks_or_cautions"])
    print(f"CONFIDENCE: {r['confidence']['label']} ({r['confidence']['score']})\n")
    print(r["disclaimer"])
