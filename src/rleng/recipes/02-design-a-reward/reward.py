"""YOUR REWARD FUNCTION. This is the file you edit in this recipe.

reward(output, expected) -> float in [0, 1]
  output:   the raw text a model produced for a task
  expected: {"tool": "get_weather", "args": {"city": "Paris"}}

The starting version below is the kind of reward people write first: "does the
output mention the right things?" It looks reasonable. Run `rleng run` and see
what an optimizer does with it.
"""


def reward(output: str, expected: dict) -> float:
    score = 0.0
    if expected["tool"] in output:
        score += 0.5
    args = expected["args"]
    for value in args.values():
        if str(value) in output:
            score += 0.5 / len(args)
    return score
