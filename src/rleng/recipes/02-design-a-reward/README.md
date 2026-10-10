# 2. Design a reward that can't be gamed

**CPU only · ~15 min · you edit one file: `reward.py`**

In RL, the policy doesn't do what you meant. It does whatever your reward pays for. Before you train
anything (recipe 4), you need a reward that pays for the right thing, and *only* the right thing.

## The task
A model gets a request like *"Convert 75 USD to CAD."* and must answer with one tool call:
```json
{"tool": "convert_currency", "args": {"amount": 75, "from_currency": "USD", "to_currency": "CAD"}}
```
`reward.py` scores the model's raw text from 0 to 1.

## Run it
```bash
rleng run       # stress test + verify
rleng inspect   # read the outputs, not just the numbers
```
`evaluate.py` builds, for 60 tasks, outputs of known kinds: correct ones, partially wrong ones, and
**hacks**, which are the kind of outputs an RL policy finds when it is pushed hard against a weak reward.
Then it simulates an optimizer: for each task, which output does your reward rank highest?

With the starting reward you should see something like:
```
hack_keyword_stuffing    1.000
simulated optimizer: picked a correct output in 0/60 tasks
```
The reward looked reasonable ("mentions the right tool and arguments"), and every hack still gets full marks.

## Your job
Change `reward.py` until `rleng verify` passes:
- correct calls score ~1.0, including harmless variations (code fences, key order, whitespace)
- every hack scores ≤ 0.1
- **partial credit**: correct > right-tool-wrong-args > wrong-tool, on every task
- never crash, never leave [0, 1]

<details><summary>Hint 1</summary>Stop matching substrings. Parse the output.</details>
<details><summary>Hint 2</summary>"Contains a valid tool call" is not the same as "is exactly one valid tool call".</details>
<details><summary>Hint 3</summary>Score the tool first, then give a fraction of the remaining credit per correct argument.</details>

## Why partial credit?
GRPO (recipe 4) samples several answers per prompt and learns from the *differences* between their
rewards. If every answer in a group gets 0, the group teaches nothing. Graded rewards keep a learning
signal even while the model is still bad.

## Go deeper
- Reward hacking examples: [Specification gaming list (DeepMind)](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- Reward design for LLM RL: [hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
