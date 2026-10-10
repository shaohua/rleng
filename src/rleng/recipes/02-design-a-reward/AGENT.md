# Agent brief: design-a-reward

**Goal:** make `reward.py` pass `rleng verify`. The starting reward looks fine but an optimizer games it.
The user is learning reward design, so the point is that they understand *why* each hack worked.

## Before you start
Ask the user one question: *"Do you want to fix the reward yourself (I'll give hints), or should I fix
it and walk you through it?"* If you can't ask, fix it yourself and explain.

## Commands
```bash
rleng run            # stress-tests reward.py, then verifies; exit 0 = pass, 1 = checks failed
rleng inspect        # mean reward per output kind + highest/lowest scored outputs
rleng inspect --json # same, machine-readable
```

## Rules
- Edit **only `reward.py`**. Changing `evaluate.py`, `tasks.py` or `recipe.yaml` to pass is reward
  hacking the grader, which is the exact failure this recipe teaches. `rleng verify` would "pass" and
  the user would learn nothing.
- No special-casing on the hack strings in `evaluate.py`. The fix must hold for any output a model could produce.

## What "done" means
`rleng run` exits 0 and all checks PASS. Then show the user:
1. The before/after table of mean reward by kind (`rleng inspect`).
2. One concrete hack the old reward loved and why the new one rejects it.
3. Why partial credit (correct > partial_args > wrong_tool) matters for GRPO in the next recipe.

## Known failure modes
| Symptom | Cause | Fix |
|---|---|---|
| `hacks_score_low` fails on `hack_call_everything` | extracting "the first `{...}`" from the text | parse the whole output; require a single object |
| `correct_scores_high` fails | rejecting ```json fences or extra whitespace | strip one surrounding fence, then `json.loads` |
| `no_crashes` fails | `json.loads` / `.get` on non-dict raises | catch errors, check types, return 0.0 |
| `partial_credit_ordering` fails | binary 0/1 reward | give credit for right tool, and per correct argument |

Optional last step: `rleng report --agent <your-name>` prints an anonymous result the user can submit.
