"""Helpers for reading agent trajectories. Use these in verifier.py.

A trajectory looks like:
  {"id": "agent-a-07", "agent": "agent-a", "task": "Convert 75 USD to CAD.",
   "expected": {"tool": "convert_currency", "args": {...}},     # the task spec
   "steps": [{"type": "tool_call", "tool": ..., "args": {...}},
             {"type": "tool_result", "content": {...}},
             {"type": "final", "content": "75 USD is about 102.75 CAD."}]}
"""

import re


def tool_calls(traj: dict) -> list[dict]:
    return [s for s in traj["steps"] if s["type"] == "tool_call"]


def tool_results(traj: dict) -> list[dict]:
    return [s["content"] for s in traj["steps"] if s["type"] == "tool_result"]


def final_answer(traj: dict) -> str:
    finals = [s["content"] for s in traj["steps"] if s["type"] == "final"]
    return finals[-1] if finals else ""


def same_args(a: dict, b: dict) -> bool:
    """Argument dicts match (case-insensitive strings, numeric tolerance)."""
    if set(a) != set(b):
        return False
    for k in a:
        try:
            if abs(float(a[k]) - float(b[k])) > 1e-9:
                return False
        except (TypeError, ValueError):
            if str(a[k]).strip().casefold() != str(b[k]).strip().casefold():
                return False
    return True


def facts(result: dict) -> list[str]:
    """The values a correct final answer must state, given a tool result. [] if the tool errored."""
    if "error" in result:
        return []
    if "temp_c" in result:
        return [f"{result['temp_c']}°C", result["condition"]]
    if "converted" in result:
        return [f"{result['converted']:.2f}"]
    if "timer_id" in result:
        return [str(result["minutes"])]
    best = min(result["flights"], key=lambda f: f["price_usd"])
    return [best["flight"], f"${best['price_usd']}"]


def grounded(answer: str, result: dict) -> bool:
    """True if the answer states every fact from the tool result (and the tool didn't error)."""
    fs = facts(result)
    return bool(fs) and all(re.search(rf"(?<![\d.]){re.escape(f)}(?!\.?\d)", answer) for f in fs)
