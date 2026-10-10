# Agent brief: eval-an-agent

**Goal:** make `verifier.py` agree with ground truth so `rleng verify` passes. The user is learning
that agent evals must check *process* and *outcome*, and that you should evaluate your evaluator.

## Before you start
Ask the user: *"Do you want to write the verifier checks yourself (I'll give hints), or should I write
them and walk you through it?"* If you can't ask, write them yourself and explain.

## Commands
```bash
rleng run            # scores 80 recorded trajectories with verifier.py, then verifies; exit 0 = pass
rleng inspect        # per failure type: how often your verifier passed it (want 0 for all but "ok")
rleng inspect --json
```

## Rules
- Edit **only `verifier.py`**. Use the helpers in `trajectory.py`.
- Never read `fixtures/labels.json` from the verifier. It is ground truth, used only to grade the grader.
  A verifier that peeks at labels "passes" and is useless in production.

## What "done" means
`rleng run` exits 0. Then show the user:
1. The table: vibe judge vs their verifier vs ground truth. The vibe judge ranks the *worse* agent higher.
2. One trajectory per failure type (`no_tool_call`, `wrong_args`, `ignored_result`, `gave_up`) and which check catches it.
3. The 95% confidence intervals: with 40 tasks per agent they are wide (±~12 points). Small evals can only detect big differences.

## Known failure modes
| Symptom | Fix |
|---|---|
| `wrong_args` still passing | compare `tool_calls(traj)[0]["args"]` with `traj["expected"]["args"]` via `same_args` |
| `ignored_result` still passing | `grounded(final_answer(traj), tool_results(traj)[0])` |
| crash on `no_tool_call` trajectories | check `tool_calls(traj)` is non-empty before indexing |
| false alarms on `ok` | don't compare raw strings; use the helpers |

Next recipe: `rleng get design-a-reward`.
