"""The task domain used across rleng recipes: pick the right tool and arguments for a user request.

The model must answer with exactly one JSON object: {"tool": <name>, "args": {...}}
"""

import json
import random

TOOLS = {
    "get_weather": {"city": "string"},
    "convert_currency": {"amount": "number", "from_currency": "string (ISO code)", "to_currency": "string (ISO code)"},
    "set_timer": {"minutes": "integer"},
    "search_flights": {"origin": "string (city)", "destination": "string (city)"},
}

SYSTEM_PROMPT = (
    "You can call exactly one tool. Tools:\n"
    + "\n".join(f"- {name}({', '.join(f'{a}: {t}' for a, t in args.items())})" for name, args in TOOLS.items())
    + '\nRespond with only a JSON object like {"tool": "<name>", "args": {...}}. No other text.'
)

CITIES = ["Paris", "Tokyo", "Lagos", "Lima", "Oslo", "Austin", "Mumbai", "Seoul", "Cairo", "Denver", "Lisbon", "Hanoi"]
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CAD", "INR", "BRL"]

TEMPLATES = {
    "get_weather": [
        "What's the weather like in {city}?",
        "Do I need an umbrella in {city} today?",
        "Is it cold in {city} right now?",
        "Weather check for {city} please.",
    ],
    "convert_currency": [
        "How much is {amount} {from_currency} in {to_currency}?",
        "Convert {amount} {from_currency} to {to_currency}.",
        "I have {amount} {from_currency}. What's that in {to_currency}?",
    ],
    "set_timer": [
        "Set a timer for {minutes} minutes.",
        "Remind me in {minutes} minutes.",
        "Start a {minutes} minute countdown.",
    ],
    "search_flights": [
        "Find flights from {origin} to {destination}.",
        "I need to fly {origin} to {destination}. What's available?",
        "Any flights {origin} -> {destination}?",
    ],
}


def _sample_args(tool: str, rng: random.Random) -> dict:
    if tool == "get_weather":
        return {"city": rng.choice(CITIES)}
    if tool == "convert_currency":
        a, b = rng.sample(CURRENCIES, 2)
        return {"amount": rng.choice([5, 20, 75, 120, 300, 1500]), "from_currency": a, "to_currency": b}
    if tool == "set_timer":
        return {"minutes": rng.choice([1, 3, 5, 10, 15, 25, 45])}
    a, b = rng.sample(CITIES, 2)
    return {"origin": a, "destination": b}


def make_tasks(n: int, seed: int = 0) -> list[dict]:
    """Return n tasks: {"prompt": str, "expected": {"tool": str, "args": dict}}."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        tool = list(TOOLS)[i % len(TOOLS)]
        args = _sample_args(tool, rng)
        prompt = rng.choice(TEMPLATES[tool]).format(**args)
        tasks.append({"prompt": prompt, "expected": {"tool": tool, "args": args}})
    rng.shuffle(tasks)
    return tasks


def to_json(call: dict) -> str:
    return json.dumps(call)
