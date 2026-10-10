# rleng: RL for engineers who ship agents

Short, hands-on recipes that take you from *"I build agents"* to *"I designed a reward, trained a
small model with GRPO, and can show it worked."* Every recipe has a machine-checkable pass/fail, so a
coding agent can run it with you.

```bash
uv tool install git+https://github.com/shaohua/rleng   # PyPI release coming: `uvx rleng`
rleng list          # the learning path
rleng get eval-an-agent && cd eval-an-agent
rleng run           # run, then verify: PASS / FAIL with hints
rleng inspect       # look at the actual outputs
```

Or tell your coding agent: **"Install rleng from github.com/shaohua/rleng and take me through the first recipe."**

No GPU? Recipes 1-2 run on any laptop. Recipe 4 runs on Apple Silicon or free Colab:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaohua/rleng/blob/main/notebooks/first-grpo-colab.ipynb)

## The path
| # | Recipe | Needs | You learn |
|---|---|---|---|
| 1 | `eval-an-agent` | CPU, 10 min | Process + outcome checks; evaluating your evaluator; a "looks good" judge prefers the worse agent |
| 2 | `design-a-reward` | CPU, 15 min | Reward hacking: watch an optimizer game a plausible reward, then fix it |
| 4 | `first-grpo` | GPU 8 GB+ or Apple Silicon, ~20 min | Train a 0.6B model with GRPO against the reward you wrote |

Recipe 3 (debug a broken run) is coming.

## Why another RL resource?
- **Verified, not aspirational.** `rleng verify` checks real metrics; hardware badges come from real runs.
- **Agent-first.** Stable CLI, JSON output, exit codes, and an `AGENT.md` brief per recipe. We measure how often coding agents finish each recipe unaided ([harness/](harness/)).
- **Inspect, don't just plot.** Every recipe ends with reading actual outputs.

For theory, use [hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/). For fast
fine-tuning notebooks, use [Unsloth](https://unsloth.ai/docs). rleng is the practical bridge for agent engineers.

## Use with Claude Code
Copy `skill/rleng/` into `~/.claude/skills/` so Claude Code knows how to drive rleng.

## Report your run
`rleng report` prints an anonymous result (no code, prompts or paths) and a prefilled GitHub issue link. Hardware reports are how recipes earn their "verified" badges.

## Development
```bash
uv venv && uv pip install -e . pytest && .venv/bin/pytest -q   # every CPU recipe must fail as shipped and pass with tests/solutions
harness/run_agent.sh design-a-reward                             # headless coding agent, clean dir, pass/fail
```
