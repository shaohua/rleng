# Agent brief: first-grpo

**Goal**: train a tiny model (Qwen3-0.6B-Base + LoRA) with GRPO so it emits correct tool calls, scored by
`reward.py`, and show that held-out accuracy went up and why.

**Done** = `rleng verify` prints `PASS` for the mode you ran. Do not edit the checks in `recipe.yaml`
to make them pass.

## Commands

```bash
rleng doctor                 # CUDA with 8GB+ -> full; Apple Silicon / CPU -> smoke (full works on MPS, just slower)
rleng run --mode smoke       # ~5 min on an M1/M2 Mac, ~2 min on a GPU. Always run this first.
rleng run --mode full        # 60 steps, ~15 min on a CUDA GPU (T4 / L4 / 4090). ~20 min on an M1 Pro.
rleng verify                 # re-check the latest run
rleng inspect -n 5           # best/worst rollouts + metrics
```

**Full mode blocks for ~16-20 minutes.** Run it in the foreground with a long command timeout (30 min).
If your tool forces you to background it, keep polling until `runs/<id>/verify.json` exists before you
finish. Ending your session early kills the training run.

`rleng run` executes, in this directory: `uv sync --locked`, `eval.py` (baseline), `train.py`,
`eval.py --adapter` (final), `summarize.py`. Each writes into `runs/<id>/`. Needs `uv` and ~5 GB of
disk for deps + model (downloaded from the HF Hub on first run).

Smoke mode only proves the pipeline works (8 steps, 24 eval tasks); accuracy is not checked (it usually rises anyway).
Full mode (60 steps, 100 eval tasks) must raise accuracy by >= 0.25 and reach >= 0.6.

## Known failure modes

| symptom | fix |
|---|---|
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then reopen the shell |
| `CUDA out of memory` / `MPS backend out of memory` | run by hand with `--num-generations 4` (the micro-batch is one group, so this halves activation memory), or close other GPU apps |
| full mode very slow on CPU | expected; use smoke mode, Apple Silicon (~16 min full), or a CUDA GPU |
| `rewards_not_degenerate` / `some_groups_have_signal` fail | every sample in a group scored the same: check `reward.py` was not replaced with a 0/1 reward; raise `--temperature 1.0` |
| `accuracy_improved` fails but `reward_gain` is fine | too few steps or unlucky run: run the steps by hand with `--steps 120` (see below), or simply rerun |
| `model_still_emits_json` fails | LR too high, policy collapsed: `--lr 2e-5` |
| HF Hub download errors / rate limits | set `HF_TOKEN`, or retry; the model is public |
| `warning: VIRTUAL_ENV=... does not match` | harmless: the recipe uses its own `.venv` |

To change hyperparameters, run the steps by hand with the env vars `rleng run` sets, then verify:

```bash
export RLENG_RUN_DIR=runs/manual RLENG_MODE=full
uv run eval.py --out $RLENG_RUN_DIR/baseline.json
uv run train.py --lr 1e-4            # any flags from `uv run train.py --help`
uv run eval.py --adapter $RLENG_RUN_DIR/adapter --out $RLENG_RUN_DIR/final.json
uv run summarize.py && rleng verify runs/manual
```

## What to show the user afterwards

1. The `rleng verify` table, and `baseline_accuracy -> final_accuracy` from `metrics.json`.
2. `reward_first -> reward_last` and `zero_std_frac_last`, with one sentence on what they mean.
3. Two rollouts from `rleng inspect`: one low-reward early output and one high-reward late output for a
   similar prompt, so the user sees the behavior change, not just the number.
4. Point them at "Knobs to try" in README.md (group size, temperature, LR) as the next experiment.
