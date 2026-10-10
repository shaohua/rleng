"""Reference reward: strict format, partial credit for arguments, never crashes."""

import json


def _parse(output: str):
    text = output.strip()
    if text.startswith("```"):
        if not text.endswith("```") or text.count("```") != 2:
            return None                 # nothing may follow the closing fence
        text = text[3:-3].removeprefix("json")
    try:
        call = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(call, dict) or set(call) != {"tool", "args"}:
        return None
    if not isinstance(call["tool"], str) or not isinstance(call["args"], dict):
        return None
    return call


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a).strip().casefold() == str(b).strip().casefold()


def reward(output: str, expected: dict) -> float:
    call = _parse(output)
    if call is None:
        return 0.0                      # not exactly one well-formed call
    if call["tool"] != expected["tool"]:
        return 0.1                      # right format, wrong tool
    want = expected["args"]
    hits = sum(k in call["args"] and _same(call["args"][k], v) for k, v in want.items())
    extra = len(set(call["args"]) - set(want))
    score = 0.4 + 0.6 * hits / len(want) - 0.1 * extra
    return max(0.0, min(1.0, score))
