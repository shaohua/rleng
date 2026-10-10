#!/usr/bin/env bash
# Run a headless coding agent on one recipe from a clean directory; print pass/fail.
# usage: AGENT=claude|codex harness/run_agent.sh <recipe-id> [trial-name]
set -uo pipefail
RECIPE=$1; TRIAL=${2:-1}; AGENT=${AGENT:-claude}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d /tmp/rleng-harness-XXXX)
export PATH="$ROOT/.venv/bin:$PATH" RLENG_HOME="$WORK/.rleng-home"
cd "$WORK"
PROMPT="The rleng CLI is installed. Use it to complete the rleng recipe '$RECIPE' end to end. Work in this directory. You are running unattended: nobody can answer questions."
# GPU recipes: unattended runs use smoke mode (headless sessions can't wait out a 20-min background job)
if grep -q "gpu: true" "$ROOT"/src/rleng/recipes/*-"$RECIPE"/recipe.yaml; then
  PROMPT="$PROMPT Use smoke mode (rleng run --mode smoke) and run it in the foreground."
fi
START=$(date +%s)
if [ "$AGENT" = codex ]; then
  # clean Codex home: no personal config/instructions, reuse the existing login
  export CODEX_HOME="$WORK/.codex-home"; mkdir -p "$CODEX_HOME"; ln -s "$HOME/.codex/auth.json" "$CODEX_HOME/auth.json"
  codex exec -s workspace-write --skip-git-repo-check -o "$WORK/agent.txt" "$PROMPT" < /dev/null > "$WORK/agent.log" 2>&1
else
  claude -p "$PROMPT" --dangerously-skip-permissions --output-format json > "$WORK/agent.json" 2> "$WORK/agent.err"
fi
END=$(date +%s)
RUN=$(ls -d "$WORK/$RECIPE"/runs/*/ 2>/dev/null | sort | tail -1)
PASS=$(python3 -c "import json,sys;print(json.load(open('${RUN}verify.json'))['pass'])" 2>/dev/null || echo "no-run")
SRC=$(ls -d "$ROOT"/src/rleng/recipes/*-"$RECIPE")
CHANGED=$(diff -rq -x runs -x __pycache__ -x .venv "$SRC" "$WORK/$RECIPE" 2>/dev/null | sed -E 's/^Files .*\/([^/]+) and .*/\1/;s/^Only in .*: /+/' | tr '\n' ' ')
python3 - "$WORK" "$RECIPE" "$TRIAL" "$PASS" $((END-START)) "$CHANGED" "$AGENT" <<'PY'
import json, sys
work, recipe, trial, passed, secs, changed, agent = sys.argv[1:]
try:
    a = json.load(open(f"{work}/agent.json"))
except Exception:
    a = {}
print(json.dumps({"agent": agent, "recipe": recipe, "trial": trial, "pass": passed, "seconds": int(secs),
                  "changed": changed.split(), "turns": a.get("num_turns"), "cost_usd": a.get("total_cost_usd"), "workdir": work}))
PY
