"""YOUR VERIFIER. This is the file you edit in this recipe.

check(traj) -> (passed: bool, reason: str)

The starting version is roughly what an LLM-as-judge "does this look like a good answer?"
check gives you. Run `rleng run` to see how well it agrees with the ground truth.
Helpers you'll want: see trajectory.py.
"""

from trajectory import final_answer, grounded, same_args, tool_calls, tool_results  # noqa: F401


def check(traj: dict) -> tuple[bool, str]:
    answer = final_answer(traj)
    if not answer:
        return False, "no final answer"
    if any(w in answer.lower() for w in ("sorry", "couldn't", "unable")):
        return False, "agent gave up"

    # TODO (process): did the agent call the expected tool with the expected args?
    #   traj["expected"] has {"tool": ..., "args": ...}

    # TODO (outcome): is the final answer grounded in what the tool actually returned?

    return True, "looks good"
