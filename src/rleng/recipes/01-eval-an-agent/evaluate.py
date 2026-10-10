"""Score two agents' recorded trajectories with your verifier, then check the verifier itself.

Ground-truth labels (fixtures/labels.json) are used ONLY here, to measure how often
your verifier agrees with reality. Your verifier must never read them.
Writes metrics.json and rollouts.jsonl into $RLENG_RUN_DIR.
"""

import json
import math
import os
import traceback
from pathlib import Path

from trajectory import final_answer
from verifier import check

run_dir = Path(os.environ.get("RLENG_RUN_DIR", "runs/manual"))
run_dir.mkdir(parents=True, exist_ok=True)
trajs = [json.loads(l) for l in open("fixtures/trajectories.jsonl")]
labels = json.load(open("fixtures/labels.json"))


def vibe_judge(traj):
    """A typical 'does it look right?' judge: confident answer that mentions the subject."""
    ans, args = final_answer(traj), traj["expected"]["args"]
    mentions = any(str(v).lower() in ans.lower() for v in args.values())
    return bool(ans) and mentions and "sorry" not in ans.lower()


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


rows, errors = [], 0
for t in trajs:
    try:
        ok, reason = check(t)
    except Exception:
        errors += 1
        if errors == 1:
            traceback.print_exc()
        ok, reason = False, "verifier crashed"
    truth = labels[t["id"]]
    rows.append({"id": t["id"], "agent": t["agent"], "kind": truth, "reward": float(bool(ok)), "reason": reason,
                 "vibe": vibe_judge(t), "prompt": t["task"],
                 "output": f"{final_answer(t)}  || calls={[(s['tool'], s['args']) for s in t['steps'] if s['type'] == 'tool_call']}"})

failures = [r for r in rows if r["kind"] != "ok"]
flagged = [r for r in rows if not r["reward"]]
tp = sum(r["kind"] != "ok" for r in flagged)
agents = sorted({r["agent"] for r in rows})
rate = lambda key, a: sum(r[key] if key != "true" else r["kind"] == "ok" for r in rows if r["agent"] == a) / sum(r["agent"] == a for r in rows)

metrics = {"verifier_recall": round(tp / len(failures), 3),
           "verifier_precision": round(tp / len(flagged), 3) if flagged else 0.0,
           "verifier_errors": errors}
for a in agents:
    k = a.split("-")[1]
    n = sum(r["agent"] == a for r in rows)
    metrics[f"vibe_pass_{k}"] = round(rate("vibe", a), 3)
    metrics[f"verifier_pass_{k}"] = round(rate("reward", a), 3)
    metrics[f"verifier_ci95_{k}"] = wilson(round(rate("reward", a) * n), n)
metrics["ranks_a_above_b"] = int(metrics["verifier_pass_a"] > metrics["verifier_pass_b"])

(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
with open(run_dir / "rollouts.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"{'':<12}{'vibe judge':>12}{'your verifier':>16}{'95% CI':>18}{'ground truth':>15}")
for a in agents:
    k = a.split("-")[1]
    print(f"{a:<12}{metrics[f'vibe_pass_{k}']:>12.0%}{metrics[f'verifier_pass_{k}']:>16.0%}{str(metrics[f'verifier_ci95_{k}']):>18}{rate('true', a):>15.0%}")
print(f"\nyour verifier vs ground truth: caught {tp}/{len(failures)} real failures (recall {metrics['verifier_recall']:.0%}), "
      f"{len(flagged) - tp} false alarms (precision {metrics['verifier_precision']:.0%})")
missed = {}
for r in failures:
    if r["reward"]:
        missed[r["kind"]] = missed.get(r["kind"], 0) + 1
if missed:
    print("missed failures by type:", missed)
    ex = next(r for r in failures if r["reward"])
    print(f"example you passed but shouldn't have [{ex['kind']}]:\n  task:   {ex['prompt']}\n  agent:  {ex['output']}")
