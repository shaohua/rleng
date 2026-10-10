---
name: rleng
description: Guide the user through rleng, short verified recipes for agent evals, reward design and small-scale GRPO training. Use when the user wants to learn or practice RL for LLM agents, reward/verifier design, evals, GRPO, or post-training on a small GPU, or mentions rleng.
---

# rleng

`rleng` is a CLI of short, hands-on RL recipes. Each one has a single clear pass/fail: `rleng verify`.

## Workflow
0. If `rleng` is not on PATH: `uv tool install git+https://github.com/shaohua/rleng`.
1. `rleng doctor --json`: what this machine can run (CPU recipes run anywhere; GPU recipes need CUDA for full mode).
2. `rleng list --json`: the learning path, in order, with `passed` status. Suggest the first recipe not yet passed.
3. `rleng get <id>`, then `cd <id>` and **read `AGENT.md` before doing anything else**. It holds the rules and the definition of done.
4. `rleng run` (add `--mode smoke` for a quick GPU pipeline check). Exit 0 = all checks pass, 1 = checks failed (read the hints), 2 = a step crashed or the environment is wrong, 3 = usage error.
5. `rleng inspect`: always show the user real outputs, not just the metrics.
6. Optional: `rleng report --agent claude-code` prints an anonymous result the user can choose to submit.

## Rules
- These are learning exercises. Ask whether the user wants to do the edit themselves (you give hints) or wants you to do it and explain.
- Only edit the file named in `AGENT.md`. Changing evaluators, fixtures or `recipe.yaml` to pass is reward hacking the grader.
- When done, explain *why* it passed, using `rleng inspect` output.
