"""Generate recorded agent trajectories for recipe 01-eval-an-agent (maintainers only).

Two simulated agents on the same 40 tasks:
  agent-a: mostly right; when a tool fails it says so (honest, sounds worse)
  agent-b: fluent and confident; skips tools, ignores results, wrong args (sounds better)
Ground-truth labels go to labels.json, which the learner's verifier must not read.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src/rleng/recipes/02-design-a-reward"))
from tasks import CITIES, CURRENCIES, make_tasks  # noqa: E402

OUT = Path(__file__).parents[1] / "src/rleng/recipes/01-eval-an-agent/fixtures"
rng = random.Random(7)
CONDITIONS = ["sunny", "light rain", "overcast", "snow", "windy", "clear"]


def run_tool(tool, args):
    if tool == "get_weather":
        return {"city": args["city"], "temp_c": rng.randint(-5, 34), "condition": rng.choice(CONDITIONS)}
    if tool == "convert_currency":
        rate = rng.uniform(0.005, 160) if "JPY" in (args["from_currency"], args["to_currency"]) else rng.uniform(0.2, 6)
        return {"amount": args["amount"], "from": args["from_currency"], "to": args["to_currency"],
                "converted": round(args["amount"] * rate, 2)}
    if tool == "set_timer":
        return {"timer_id": f"tmr_{rng.randint(1000, 9999)}", "minutes": args["minutes"], "status": "running"}
    flights = [{"flight": f"{rng.choice('QAZXLM')}{rng.choice('QAZXLM')}{rng.randint(100, 999)}",
                "price_usd": rng.randint(90, 900)} for _ in range(3)]
    return {"origin": args["origin"], "destination": args["destination"], "flights": flights}


def answer(tool, args, res, fudge=False):
    """Fluent final answer grounded in `res` (or subtly not, if fudge)."""
    if tool == "get_weather":
        t = res["temp_c"] + (rng.choice([-6, -4, 5, 7]) if fudge else 0)
        cond = rng.choice([c for c in CONDITIONS if c != res["condition"]]) if fudge else res["condition"]
        return f"Right now in {args['city']} it's {t}°C with {cond}."
    if tool == "convert_currency":
        v = res["converted"] * (rng.choice([0.9, 1.12, 0.8]) if fudge else 1)
        return f"{args['amount']} {args['from_currency']} is about {v:.2f} {args['to_currency']}."
    if tool == "set_timer":
        m = args["minutes"] + (rng.choice([5, 10]) if fudge else 0)
        return f"Done! Your {m}-minute timer is running."
    best = min(res["flights"], key=lambda f: f["price_usd"])
    if fudge:
        best = {"flight": best["flight"], "price_usd": best["price_usd"] - rng.randint(20, 60)}
    return f"The cheapest option from {args['origin']} to {args['destination']} is {best['flight']} at ${best['price_usd']}."


def hallucinated_result(tool, args):
    return run_tool(tool, args)  # what the agent *imagines* the tool would say


def corrupt_args(tool, args):
    a = dict(args)
    if tool == "get_weather":
        a["city"] = rng.choice([c for c in CITIES if c != args["city"]])
    elif tool == "convert_currency":
        a["from_currency"], a["to_currency"] = args["to_currency"], args["from_currency"]
    elif tool == "set_timer":
        a["minutes"] = args["minutes"] * 60  # minutes vs seconds confusion
    else:
        a["origin"], a["destination"] = args["destination"], args["origin"]
    return a


def trajectory(task, mode):
    tool, args = task["expected"]["tool"], task["expected"]["args"]
    steps = []
    if mode == "no_tool_call":
        steps.append({"type": "final", "content": answer(tool, args, hallucinated_result(tool, args))})
        return steps
    call_args = corrupt_args(tool, args) if mode == "wrong_args" else args
    steps.append({"type": "tool_call", "tool": tool, "args": call_args})
    if mode == "gave_up":
        steps.append({"type": "tool_result", "content": {"error": "upstream timeout"}})
        steps.append({"type": "final", "content": "Sorry, the service timed out and I couldn't complete that. Want me to try again?"})
        return steps
    res = run_tool(tool, call_args)
    steps.append({"type": "tool_result", "content": res})
    final_args = args if mode == "wrong_args" else call_args  # wrong_args: answer still names what the user asked for
    steps.append({"type": "final", "content": answer(tool, final_args, res, fudge=(mode == "ignored_result"))})
    return steps


tasks = make_tasks(40, seed=123)
plans = {
    "agent-a": ["ok"] * 33 + ["gave_up"] * 4 + ["wrong_args"] * 2 + ["ignored_result"] * 1,
    "agent-b": ["ok"] * 25 + ["no_tool_call"] * 7 + ["ignored_result"] * 5 + ["wrong_args"] * 3,
}
rows, labels = [], {}
for agent, plan in plans.items():
    plan = plan[:]
    rng.shuffle(plan)
    for i, (task, mode) in enumerate(zip(tasks, plan)):
        tid = f"{agent}-{i:02d}"
        rows.append({"id": tid, "agent": agent, "task": task["prompt"], "expected": task["expected"],
                     "steps": trajectory(task, mode)})
        labels[tid] = mode
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "trajectories.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
(OUT / "labels.json").write_text(json.dumps(labels, indent=1))
print(f"wrote {len(rows)} trajectories to {OUT}")
