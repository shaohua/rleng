# Your first GRPO run

You built a reward in recipe 2. Now you use it to train a model. A tiny model (Qwen3-0.6B-Base) gets
the same tool-calling tasks; at the start it gets ~20% right (it often knows the call but wraps it
in junk like `[]{...}`, or picks the wrong tool). After a few dozen GRPO steps with a small LoRA
adapter it gets nearly all of them right, and the rollouts show you how it got there.

(Why a base model? The instruct versions of Qwen3-0.6B and Qwen2.5-0.5B already score 90-100% on
these tasks, so there is nothing left to learn.)

```bash
rleng get first-grpo && cd first-grpo
rleng run --mode smoke   # ~5 min, any machine (Apple Silicon ok): pipeline works end to end
rleng run --mode full    # 60 steps: ~16 min on an M1 Pro (verified); CUDA T4/24GB target, not yet verified
rleng inspect            # look at what the model actually did
```

## What GRPO is doing here

1. Take a batch of tasks ("Convert 20 EUR to JPY.").
2. For each task, sample a **group** of 8 completions from the current model (temperature 0.8).
3. Score each completion with `reward.py` (your recipe-2 reward: 0 for unparseable, 0.1 for the wrong
   tool, 0.4-1.0 for the right tool depending on how many args are right).
4. **Advantage** = how much better a completion scored than the average of its own group. No value
   network, no critic: the group is the baseline.
5. Raise the log-probability of tokens in above-average completions, lower it for below-average ones
   (clipped, like PPO). Only the LoRA adapter (~1% of weights) is trained.
6. If all 8 completions in a group get the same reward, every advantage is 0 and that task teaches
   nothing. That is why partial credit matters, and why we log `zero_std_frac`.

## What the run produces (`runs/<id>/`)

| file | what |
|---|---|
| `baseline.json`, `final.json` | greedy decode on 100 held-out tasks, before and after training. Held-out means *new entities*: eval tasks use cities, currencies and timer lengths that never appear in training, so a high score means the model generalized, not memorized |
| `train_log.jsonl` | per step: mean reward, reward std, fraction of zero-std groups, completion length |
| `rollouts.jsonl` | every sampled completion: `{step, prompt, output, reward}` |
| `metrics.json` | what `rleng verify` checks |
| `adapter/` | the trained LoRA weights |

## What to look at in `rleng inspect`

- **Lowest-reward rollouts**: early on, prose, a `[]` prefix before the JSON, or the wrong tool.
  These are what the model is being pushed away from.
- **Highest-reward rollouts**: should be clean JSON calls. If high-reward outputs look wrong to you,
  the reward is being hacked: go back to recipe 2.
- `reward_first` vs `reward_last`, and `zero_std_frac_last`: near 1.0 with high reward means the
  model has solved the training tasks (groups all score 1.0); near 1.0 with low reward means it is
  stuck with no signal.
- Compare a task's rollouts across steps:
  `grep '"Set a timer' runs/<id>/rollouts.jsonl | head`.

## Our verified run (M1 Pro 16 GB, full mode)

Held-out accuracy 0.17 -> 1.00, valid JSON 0.27 -> 1.00, train reward 0.85 -> 1.00, 943 s of training,
6.3 GB peak. Note `zero_std_frac_mean` = 0.90: after ~15 steps nearly every group scores 1.0 on every
sample, so most of the run teaches nothing. This task is easy; a harder multi-step recipe is planned.

## Knobs to try (`uv run train.py --help`)

Untested predictions; check them against `train_log.jsonl` and `rleng inspect`.

| flag | default | try | what to watch |
|---|---|---|---|
| `--num-generations` | 8 | 2, 16 | G=2: more zero-std groups, noisier reward curve. G=16: smoother, slower steps |
| `--lr` | 5e-5 | 1e-5, 2e-4 | too low: reward rises slowly. too high: watch for broken JSON / falling reward |
| `--temperature` | 0.8 | 0.3, 1.2 | low: groups more identical (less signal). high: more garbage samples |
| `--beta` | 0.0 | 0.05 | KL penalty to the base model: slower drift away from it, extra memory for the reference |
| `--steps` | 60 | 20 | in our runs most of the gain came in the first ~20 steps; after that groups are all 1.0 |

Want a slower, more interesting learning curve? `RLENG_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct rleng run`
(baseline ~27%, ~70% after 30 steps; its mistakes are wrong-tool choices, not formatting).

Then swap `reward.py` for the naive reward from recipe 2 and see what the model learns to do.

## Go deeper

- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
- Hands-on Modern RL (policy gradients, PPO, GRPO): https://walkinglabs.github.io/hands-on-modern-rl/
