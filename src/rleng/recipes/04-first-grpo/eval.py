"""Greedy-decode held-out tasks and score them with reward.py.

accuracy = fraction of tasks with reward == 1.0 (right tool, all args right, valid JSON).

  uv run eval.py --out runs/x/baseline.json                       # base model
  uv run eval.py --adapter runs/x/adapter --out runs/x/final.json  # base model + your LoRA
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CHAT_KWARGS, split_tasks, MAX_NEW_TOKENS, MODE, MODEL_ID, device, load_dtype, messages
from reward import reward

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--adapter", help="path to a LoRA adapter saved by train.py")
p.add_argument("--n", type=int, default=24 if MODE == "smoke" else 100, help="number of held-out tasks")
p.add_argument("--batch-size", type=int, default=16)
args = p.parse_args()

dev = device()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=load_dtype()).to(dev)
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

tasks = split_tasks(args.n, "eval")
rows, t0 = [], time.time()
for i in range(0, len(tasks), args.batch_size):
    batch = tasks[i : i + args.batch_size]
    texts = [tok.apply_chat_template(messages(t["prompt"]), tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
             for t in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
    for t, ids in zip(batch, out[:, enc["input_ids"].shape[1]:]):
        text = tok.decode(ids, skip_special_tokens=True)
        rows.append({"prompt": t["prompt"], "expected": t["expected"], "output": text, "reward": reward(text, t["expected"])})

result = {
    "model": MODEL_ID,
    "adapter": args.adapter,
    "device": dev,
    "n": len(rows),
    "accuracy": round(sum(r["reward"] == 1.0 for r in rows) / len(rows), 3),
    "mean_reward": round(sum(r["reward"] for r in rows) / len(rows), 3),
    "valid_json": round(sum(r["reward"] > 0 for r in rows) / len(rows), 3),
    "seconds": round(time.time() - t0, 1),
    "rows": rows,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=2))
print(f"{'final' if args.adapter else 'baseline'}: accuracy={result['accuracy']} mean_reward={result['mean_reward']} "
      f"valid_json={result['valid_json']} n={result['n']} ({result['seconds']}s on {dev})")
for r in [r for r in rows if r["reward"] < 1.0][:3]:
    print(f"  miss ({r['reward']:.2f}) {r['prompt']!r} -> {r['output'][:120]!r}")
