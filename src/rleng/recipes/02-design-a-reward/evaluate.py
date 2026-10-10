"""Stress-test reward.py against honest outputs and outputs that try to game it.

For every task we build candidate outputs of known kinds:
  correct         right tool, right args (a few harmless format variants)
  partial_args    right tool, one argument wrong
  wrong_tool      valid JSON, wrong tool
  hack_*          outputs a policy under optimization pressure tends to find

Then we simulate an optimizer: for each task, which candidate does your reward
rank highest? (Ties go to the non-correct candidate, because an RL policy will
find whatever the reward can't tell apart.)

Writes metrics.json and rollouts.jsonl into $RLENG_RUN_DIR.
"""

import json
import os
import random
import traceback
from pathlib import Path

from reward import reward
from tasks import CITIES, TOOLS, _sample_args, make_tasks, to_json

N_TASKS = 60
run_dir = Path(os.environ.get("RLENG_RUN_DIR", "runs/manual"))
run_dir.mkdir(parents=True, exist_ok=True)
rng = random.Random(1)


def candidates(task: dict) -> list[tuple[str, str]]:
    exp = task["expected"]
    tool, args = exp["tool"], exp["args"]
    out = []

    # honest, correct (harmless format variants a real model produces)
    variant = rng.choice(["plain", "fenced", "reordered", "spaced"])
    if variant == "plain":
        out.append(("correct", to_json(exp)))
    elif variant == "fenced":
        out.append(("correct", "```json\n" + to_json(exp) + "\n```"))
    elif variant == "reordered":
        out.append(("correct", json.dumps({"args": args, "tool": tool})))
    else:
        out.append(("correct", "  " + json.dumps(exp, indent=2) + "\n"))

    # right tool, one wrong argument
    bad_args = dict(args)
    key = rng.choice(list(args))
    while bad_args[key] == args[key]:
        bad_args[key] = _sample_args(tool, rng).get(key, rng.choice(CITIES))
    out.append(("partial_args", to_json({"tool": tool, "args": bad_args})))

    # valid format, wrong tool
    other = rng.choice([t for t in TOOLS if t != tool])
    out.append(("wrong_tool", to_json({"tool": other, "args": _sample_args(other, rng)})))

    # hacks
    out.append(("hack_keyword_stuffing", " ".join(TOOLS) + " " + task["prompt"]))
    out.append(("hack_not_json", f"tool={tool} args: " + ", ".join(f"{k}={v}" for k, v in args.items())))
    calls = [{"tool": tool, "args": args}] + [{"tool": t, "args": _sample_args(t, rng)} for t in TOOLS if t != tool]
    rng.shuffle(calls)
    out.append(("hack_call_everything", json.dumps(calls)))
    out.append(("hack_trailing_junk", "```json\n" + to_json(exp) + "\n```\n" + " ".join(t for t in TOOLS if t != tool)))
    out.append(("hack_prose", f"Sure! I would use {tool} with " + " and ".join(str(v) for v in args.values()) + ". Let me know if that helps!"))
    return out


rows, errors = [], 0
for i, task in enumerate(make_tasks(N_TASKS, seed=42)):
    for kind, output in candidates(task):
        try:
            r = float(reward(output, task["expected"]))
        except Exception:
            errors += 1
            if errors == 1:
                traceback.print_exc()
            r = 0.0
        rows.append({"task": i, "prompt": task["prompt"], "expected": task["expected"], "kind": kind, "output": output, "reward": r})

by_kind: dict[str, list[float]] = {}
for row in rows:
    by_kind.setdefault(row["kind"], []).append(row["reward"])
mean = {k: sum(v) / len(v) for k, v in by_kind.items()}

# per-task ordering and optimizer simulation
ordered, picked_correct, picks = 0, 0, {}
for i in range(N_TASKS):
    task_rows = {r["kind"]: r for r in rows if r["task"] == i}
    c, p, w = (task_rows[k]["reward"] for k in ("correct", "partial_args", "wrong_tool"))
    ordered += c > p > w
    best = max(task_rows.values(), key=lambda r: (r["reward"], r["kind"] != "correct"))
    picks[best["kind"]] = picks.get(best["kind"], 0) + 1
    picked_correct += best["kind"] == "correct"

hack_rewards = [r["reward"] for r in rows if r["kind"].startswith("hack_")]
metrics = {
    "correct_mean": round(mean["correct"], 3),
    "hack_max": round(max(hack_rewards), 3),
    "ordering_ok_rate": round(ordered / N_TASKS, 3),
    "optimizer_picks_correct": round(picked_correct / N_TASKS, 3),
    "out_of_range": sum(not (0.0 <= r["reward"] <= 1.0) for r in rows),
    "reward_errors": errors,
}

(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
with open(run_dir / "rollouts.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print("mean reward by kind (want: correct ~1, partial in between, wrong_tool low, hacks ~0)")
for k in ["correct", "partial_args", "wrong_tool"] + sorted(k for k in mean if k.startswith("hack_")):
    print(f"  {k:<24} {mean[k]:.3f}")
print(f"\nsimulated optimizer: picked a correct output in {picked_correct}/{N_TASKS} tasks")
print("  what it picked instead:", {k: v for k, v in sorted(picks.items()) if k != "correct"} or "nothing - nice")
worst = max((r for r in rows if r["kind"].startswith("hack_")), key=lambda r: r["reward"])
if worst["reward"] > 0.1:
    print(f"\nexample hack your reward likes ({worst['reward']:.2f}):\n  prompt:   {worst['prompt']}\n  output:   {worst['output']}")
