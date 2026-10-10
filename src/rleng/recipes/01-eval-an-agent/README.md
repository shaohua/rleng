# 1. Evaluate an agent (and your evaluator)

**CPU only · ~10 min · you edit one file: `verifier.py`**

Before you can improve an agent, with prompts or with RL, you need a score you can trust.
Here you have 80 recorded runs of two tool-using agents on the same 40 tasks (weather, currency,
timers, flights). Your job: write a verifier that says which runs actually succeeded.

## Run it
```bash
rleng run       # score both agents, compare to ground truth
rleng inspect   # read the trajectories your verifier got wrong
```
With the starting verifier (roughly what a "does this answer look good?" judge does):
```
              vibe judge   your verifier   ground truth
agent-a              88%             90%            82%
agent-b              92%            100%            62%
```
Agent B sounds confident and fluent, but it skips tools, ignores tool results, and passes the wrong
arguments. Agent A sometimes admits a tool failed. A judge that rewards sounding right picks the worse agent.

## Your job
Make the verifier catch real failures (recall ≥ 95%) without false alarms (precision ≥ 95%):
1. **Process:** did the agent call the expected tool with the expected arguments?
2. **Outcome:** is the final answer grounded in what the tool actually returned?

Helpers in `trajectory.py`: `tool_calls`, `tool_results`, `final_answer`, `same_args`, `grounded`.

## Takeaways
- Evaluate the evaluator. A small labeled set (even 50 hand-labeled runs) tells you if your metric means anything.
- Check process *and* outcome. Each one alone misses a whole class of failures.
- Report uncertainty. 40 tasks gives ±~12 points; don't ship a "3% better" claim from that.
- The verifier you just wrote is a **reward function**. Recipe 2 asks what happens when something optimizes against it.

## Go deeper
- [hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/): the reward and evaluation chapters
- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
