"""Shared settings for train.py and eval.py: model, device, prompt format."""

import os

import torch

from tasks import SYSTEM_PROMPT, make_tasks

# Base, not instruct: Qwen3-0.6B (instruct) already scores ~100% on this task, the base model ~18%.
MODEL_ID = os.environ.get("RLENG_MODEL", "Qwen/Qwen3-0.6B-Base")
MAX_NEW_TOKENS = 64
TRAIN_SEED, EVAL_SEED = 7, 1234
# Held-out entities: eval only uses tasks containing one of these, training never sees them.
# So eval measures generalization to new cities/currencies/durations, not memorized prompts.
HELDOUT = {"Oslo", "Hanoi", "Lima", "BRL", 25, 45}
CHAT_KWARGS = {"enable_thinking": False}  # Qwen3 template: no <think> block; other templates ignore it
MODE = os.environ.get("RLENG_MODE", "full")


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dtype() -> torch.dtype:
    """bf16 where the GPU supports it; otherwise fp32 weights (T4 then trains with fp16 autocast)."""
    if device() == "cuda" and torch.cuda.get_device_capability()[0] >= 8:  # Ampere+; T4 is 7.5
        return torch.bfloat16
    return torch.float32


def split_tasks(n: int, split: str) -> list[dict]:
    """n tasks for 'train' (no held-out entities) or 'eval' (at least one held-out entity)."""
    want_heldout = split == "eval"
    pool = make_tasks(20 * n + 200, seed=EVAL_SEED if want_heldout else TRAIN_SEED)
    out = [t for t in pool if any(v in HELDOUT for v in t["expected"]["args"].values()) == want_heldout]
    return out[:n]


def messages(prompt: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
