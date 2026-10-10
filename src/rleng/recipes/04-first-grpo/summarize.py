"""Merge baseline.json, final.json and train_log.jsonl into $RLENG_RUN_DIR/metrics.json (what `rleng verify` checks)."""

import json
import math
import os
from pathlib import Path

run_dir = Path(os.environ.get("RLENG_RUN_DIR", "runs/manual"))
baseline = json.loads((run_dir / "baseline.json").read_text())
final = json.loads((run_dir / "final.json").read_text())
log = [json.loads(line) for line in (run_dir / "train_log.jsonl").open()]
info = json.loads((run_dir / "train_info.json").read_text())
n_rollouts = sum(1 for _ in (run_dir / "rollouts.jsonl").open())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


k = max(1, math.ceil(0.2 * len(log)))  # first / last 20% of steps
first, last = log[:k], log[-k:]
metrics = {
    "baseline_accuracy": baseline["accuracy"],
    "final_accuracy": final["accuracy"],
    "accuracy_gain": round(final["accuracy"] - baseline["accuracy"], 3),
    "baseline_valid_json": baseline["valid_json"],
    "final_valid_json": final["valid_json"],
    "eval_tasks": final["n"],
    "train_steps": len(log),
    "reward_first": mean(r["reward"] for r in first),
    "reward_last": mean(r["reward"] for r in last),
    "reward_std_mean": mean(r["reward_std"] for r in log),
    "zero_std_frac_mean": mean(r["frac_zero_std"] for r in log),
    "zero_std_frac_last": mean(r["frac_zero_std"] for r in last),
    "completion_len_last": mean(r["completion_len"] for r in last),
    "rollouts": n_rollouts,
    "train_seconds": log[-1]["seconds"],
    "peak_mem_gb": info["peak_mem_gb"],
    "device": info["device"],
    "model": info["model"],
}
metrics["reward_gain"] = round(metrics["reward_last"] - metrics["reward_first"], 3)
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

print(f"eval accuracy  {metrics['baseline_accuracy']:.2f} -> {metrics['final_accuracy']:.2f}  "
      f"(gain {metrics['accuracy_gain']:+.2f}, n={metrics['eval_tasks']})")
print(f"train reward   {metrics['reward_first']:.2f} -> {metrics['reward_last']:.2f}  "
      f"(first/last {k} of {len(log)} steps)")
print(f"zero-std groups in last steps: {metrics['zero_std_frac_last']:.2f}   rollouts saved: {n_rollouts}")
