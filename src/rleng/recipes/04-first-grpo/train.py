"""GRPO + LoRA: teach a tiny model to emit correct tool calls, scored by reward.py.

Each step: take PROMPTS_PER_STEP tasks, sample NUM_GENERATIONS completions for each, score them with
reward.py, and push the policy toward completions that beat their group's average.

Writes into $RLENG_RUN_DIR:
  train_log.jsonl   one row per step: mean reward, reward std, fraction of zero-std groups, ...
  rollouts.jsonl    every sampled completion: {"step", "prompt", "output", "reward"}
  adapter/          the trained LoRA adapter (eval.py --adapter loads it)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from common import CHAT_KWARGS, MAX_NEW_TOKENS, MODE, MODEL_ID, TRAIN_SEED, device, split_tasks, load_dtype, messages
from reward import reward

# ---- knobs (see README "Knobs to try") ----
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=8 if MODE == "smoke" else 60)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--num-generations", type=int, default=8, help="group size G: completions sampled per prompt")
p.add_argument("--prompts-per-step", type=int, default=4)
p.add_argument("--beta", type=float, default=0.0, help="KL penalty to the base model (0 = no reference model)")
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--lora-r", type=int, default=16)
args = p.parse_args()

run_dir = Path(os.environ.get("RLENG_RUN_DIR", "runs/manual"))
run_dir.mkdir(parents=True, exist_ok=True)
rollouts_path, log_path = run_dir / "rollouts.jsonl", run_dir / "train_log.jsonl"
rollouts_path.unlink(missing_ok=True)
log_path.unlink(missing_ok=True)

# ---- data: prompts in chat format, expected call kept as a JSON string column ----
n_tasks = args.steps * args.prompts_per_step
tasks = split_tasks(n_tasks, "train")
dataset = Dataset.from_list([{"prompt": messages(t["prompt"]), "expected": json.dumps(t["expected"])} for t in tasks])


# ---- reward: thin wrapper that adapts reward.py to TRL's signature and records every rollout ----
def tool_call_reward(prompts, completions, expected, trainer_state, **kwargs):
    texts = [c[0]["content"] for c in completions]
    scores = [float(reward(text, json.loads(exp))) for text, exp in zip(texts, expected)]
    with rollouts_path.open("a") as f:
        for prompt, text, score in zip(prompts, texts, scores):
            row = {"step": trainer_state.global_step + 1, "prompt": prompt[-1]["content"], "output": text, "reward": score}
            f.write(json.dumps(row) + "\n")
    return scores


class StepLog(TrainerCallback):
    """Append TRL's per-step metrics to train_log.jsonl and print a one-line summary."""

    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "reward" not in logs:
            return
        row = {"step": state.global_step, "reward": logs["reward"], "reward_std": logs.get("reward_std"),
               "frac_zero_std": logs.get("frac_reward_zero_std"), "completion_len": logs.get("completions/mean_length"),
               "kl": logs.get("kl"), "entropy": logs.get("entropy"), "seconds": round(time.time() - self.t0, 1)}
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {row['step']:>4}/{state.max_steps}  reward {row['reward']:.3f}  "
              f"zero-std groups {row['frac_zero_std']:.2f}  len {row['completion_len']:.0f}  {row['seconds']:.0f}s", flush=True)


# ---- model + trainer ----
dev = device()
on_cuda = dev == "cuda"
dtype = load_dtype()

config = GRPOConfig(
    output_dir=str(run_dir / "trainer"),
    max_steps=args.steps,
    learning_rate=args.lr,
    lr_scheduler_type="constant",
    warmup_steps=0,
    per_device_train_batch_size=args.num_generations,  # one group per micro-batch keeps activation memory small
    gradient_accumulation_steps=args.prompts_per_step,  # -> one optimizer step per generation batch
    num_generations=args.num_generations,
    max_completion_length=MAX_NEW_TOKENS,
    temperature=args.temperature,
    beta=args.beta,
    chat_template_kwargs=CHAT_KWARGS,
    bf16=on_cuda and dtype == torch.bfloat16,
    fp16=on_cuda and dtype != torch.bfloat16,  # T4: fp32 weights + fp16 autocast
    gradient_checkpointing=True,  # trades ~30% speed for much less activation memory (fits 16GB Macs)
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=TRAIN_SEED,
    use_cpu=dev == "cpu",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Load on CPU, then move: device_map="auto" loading can segfault on MPS with transformers 5.x
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(dev)
peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                         target_modules="all-linear", task_type="CAUSAL_LM")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=tool_call_reward,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[StepLog()],
)
print(f"GRPO: model={MODEL_ID} device={dev} dtype={dtype} steps={args.steps} "
      f"prompts/step={args.prompts_per_step} G={args.num_generations} lr={args.lr} beta={args.beta}", flush=True)
trainer.train()
trainer.model.save_pretrained(run_dir / "adapter")

peak = None
if on_cuda:
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
elif dev == "mps":
    peak = round(torch.mps.driver_allocated_memory() / 1e9, 2)
(run_dir / "train_info.json").write_text(json.dumps(
    {"model": MODEL_ID, "device": dev, "dtype": str(dtype), "peak_mem_gb": peak, **vars(args)}, indent=2))
print(f"saved adapter -> {run_dir / 'adapter'} (peak memory {peak} GB)")
