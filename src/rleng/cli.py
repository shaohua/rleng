"""rleng CLI: list, doctor, get, run, verify, inspect, report.

Exit codes: 0 ok/pass, 1 checks failed, 2 step or environment error, 3 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from datetime import datetime
from importlib import resources
from pathlib import Path

import yaml

from rleng import __version__

REPO = os.environ.get("RLENG_REPO", "shaohua/rleng")
PROGRESS_FILE = Path(os.environ.get("RLENG_HOME", Path.home() / ".rleng")) / "progress.json"
OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


class UsageError(Exception):
    pass


# ---------- helpers ----------

def emit(args, data: dict, text: str) -> None:
    print(json.dumps(data, indent=2) if args.json else text)


def recipes_root() -> Path:
    return Path(str(resources.files("rleng") / "recipes"))


def load_recipe(recipe_dir: Path) -> dict:
    path = recipe_dir / "recipe.yaml"
    if not path.exists():
        raise UsageError(f"no recipe.yaml in {recipe_dir}. Run `rleng get <id>` first, then cd into it.")
    recipe = yaml.safe_load(path.read_text())
    recipe["_dir"] = recipe_dir
    return recipe


def all_recipes() -> list[dict]:
    dirs = [d for d in recipes_root().iterdir() if (d / "recipe.yaml").exists()]
    return sorted((load_recipe(d) for d in dirs), key=lambda r: r["order"])


def find_recipe(recipe_id: str) -> dict:
    for r in all_recipes():
        if recipe_id in (r["id"], r["_dir"].name, str(r["order"])):
            return r
    ids = ", ".join(r["id"] for r in all_recipes())
    raise UsageError(f"unknown recipe '{recipe_id}'. Available: {ids}")


def latest_run(recipe_dir: Path) -> Path:
    runs = sorted((recipe_dir / "runs").glob("*/")) if (recipe_dir / "runs").exists() else []
    if not runs:
        raise UsageError(f"no runs in {recipe_dir}/runs. Run `rleng run` first.")
    return runs[-1]


def resolve_run(arg: str | None) -> Path:
    """Accept a run dir, a recipe dir (-> latest run), or nothing (cwd -> latest run)."""
    path = Path(arg or ".").resolve()
    if (path / "metrics.json").exists() or (path / "summary.json").exists():
        return path
    return latest_run(path)


def load_progress() -> dict:
    return json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}


def detect_gpu() -> dict:
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem = out.stdout.strip().splitlines()[0].split(", ")
            return {"type": "cuda", "name": name, "vram_gb": round(int(mem) / 1024, 1)}
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return {"type": "mps", "name": "Apple Silicon", "vram_gb": None}
    return {"type": "none", "name": None, "vram_gb": None}


# ---------- commands ----------

def cmd_list(args) -> int:
    progress = load_progress()
    rows = []
    for r in all_recipes():
        hw = r.get("hardware", {})
        rows.append({
            "order": r["order"],
            "id": r["id"],
            "title": r["title"],
            "gpu": bool(hw.get("gpu")),
            "min_vram_gb": hw.get("min_vram_gb"),
            "time": r.get("time", {}),
            "requires": r.get("requires", []),
            "passed": r["id"] in progress,
        })
    lines = ["Learning path (run in order):", ""]
    for x in rows:
        hw = f"GPU {x['min_vram_gb']}GB+" if x["gpu"] else "CPU"
        mark = "✓" if x["passed"] else " "
        lines.append(f" [{mark}] {x['order']}. {x['id']:<22} {hw:<10} ~{x['time'].get('full', '?'):<5} {x['title']}")
    lines += ["", "Start: rleng get <id>"]
    emit(args, {"recipes": rows}, "\n".join(lines))
    return 0


def cmd_doctor(args) -> int:
    gpu = detect_gpu()
    disk_gb = round(shutil.disk_usage(".").free / 1e9, 1)
    info = {
        "rleng": __version__,
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "uv": shutil.which("uv") is not None,
        "gpu": gpu,
        "disk_free_gb": disk_gb,
    }
    runnable, problems = [], []
    if not info["uv"]:
        problems.append("uv not found: install with `curl -LsSf https://astral.sh/uv/install.sh | sh` (GPU recipes need it)")
    for r in all_recipes():
        hw = r.get("hardware", {})
        if not hw.get("gpu"):
            runnable.append({"id": r["id"], "modes": ["smoke", "full"]})
        elif gpu["type"] == "cuda" and gpu["vram_gb"] >= hw.get("min_vram_gb", 0):
            runnable.append({"id": r["id"], "modes": ["smoke", "full"]})
        elif gpu["type"] == "mps":
            runnable.append({"id": r["id"], "modes": ["smoke", "full"], "note": "Apple Silicon: works, ~2-3x slower than a CUDA GPU"})
        elif info["uv"]:
            runnable.append({"id": r["id"], "modes": ["smoke"],
                             "note": f"CPU only: smoke mode is slow; full mode needs a GPU with {hw.get('min_vram_gb')}GB+ (e.g. Colab T4)"})
    if disk_gb < 10:
        problems.append(f"only {disk_gb}GB disk free; GPU recipes need ~10GB for model + deps")
    info["runnable"] = runnable
    info["problems"] = problems

    gpu_str = {"cuda": f"{gpu['name']} ({gpu['vram_gb']} GB)", "mps": "Apple Silicon (MPS)",
               "none": "none - CPU recipes only"}[gpu["type"]]
    lines = [f"rleng {__version__} | python {info['python']} | {info['platform']}",
             f"GPU: {gpu_str}", f"uv: {'yes' if info['uv'] else 'NO'} | disk free: {disk_gb} GB", "", "Runnable here:"]
    for x in runnable:
        lines.append(f"  {x['id']:<22} modes: {', '.join(x['modes'])}" + (f"  ({x['note']})" if x.get("note") else ""))
    lines += [f"PROBLEM: {p}" for p in problems]
    emit(args, info, "\n".join(lines))
    return 2 if problems else 0


def cmd_get(args) -> int:
    r = find_recipe(args.id)
    dest = Path(args.dir or r["id"]).resolve()
    if dest.exists() and not args.force:
        raise UsageError(f"{dest} already exists (use --force to overwrite)")
    shutil.copytree(r["_dir"], dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("runs", "__pycache__", ".venv"))
    data = {"id": r["id"], "path": str(dest), "agent_brief": str(dest / "AGENT.md"),
            "next": f"cd {dest.name} && rleng run"}
    emit(args, data, f"Fetched '{r['id']}' into {dest}\n\nNext:\n  cd {dest.name}\n  cat AGENT.md   # agents: read this first\n  rleng run")
    return 0


def cmd_run(args) -> int:
    recipe_dir = Path(args.path).resolve()
    r = load_recipe(recipe_dir)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.mode}"
    run_dir = recipe_dir / "runs" / run_id
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = recipe_dir / "runs" / f"{run_id}-{n}"
    run_dir.mkdir(parents=True)
    env = {**os.environ, "RLENG_RUN_DIR": str(run_dir), "RLENG_MODE": args.mode, "PYTHONUNBUFFERED": "1"}
    env.pop("VIRTUAL_ENV", None)  # recipes with their own pyproject use `uv run`; don't leak rleng's venv
    subs = {"python": sys.executable, "run_dir": str(run_dir), "mode": args.mode}
    summary = {"recipe": r["id"], "recipe_version": r.get("version", 1), "mode": args.mode, "run_dir": str(run_dir),
               "gpu": detect_gpu(), "steps": [], "started": datetime.now().isoformat(timespec="seconds")}
    t0 = time.time()
    with open(run_dir / "log.txt", "w") as log:
        for step in r["steps"]:
            if args.mode not in step.get("modes", ["smoke", "full"]):
                continue
            cmd = step["run"].format(**subs)
            print(f"\n=== step: {step['name']} ===\n$ {cmd}", file=sys.stderr, flush=True)
            st = time.time()
            proc = subprocess.Popen(cmd, shell=True, cwd=recipe_dir, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                sys.stderr.write(line)
                log.write(line)
            code = proc.wait()
            summary["steps"].append({"name": step["name"], "exit_code": code, "seconds": round(time.time() - st, 1)})
            if code != 0:
                summary["duration_s"] = round(time.time() - t0, 1)
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                emit(args, {**summary, "pass": False, "error": f"step '{step['name']}' failed"},
                     f"\nFAILED: step '{step['name']}' exited {code}. Full log: {run_dir / 'log.txt'}")
                return 2
    summary["duration_s"] = round(time.time() - t0, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun complete in {summary['duration_s']}s -> {run_dir}\n", file=sys.stderr)
    return verify_run(args, run_dir)


def verify_run(args, run_dir: Path) -> int:
    recipe_dir = run_dir.parent.parent
    r = load_recipe(recipe_dir)
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    mode = summary.get("mode", "full")
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    results = []
    for c in r.get("checks", []):
        if mode not in c.get("modes", ["smoke", "full"]):
            continue
        res = {"id": c["id"], "hint": c.get("hint", "")}
        if "file" in c:
            p = run_dir / c["file"]
            rows = sum(1 for _ in p.open()) if p.exists() else 0
            res.update(observed=f"{rows} rows in {c['file']}", passed=rows >= c.get("min_rows", 1))
        else:
            val = metrics.get(c["metric"])
            ok = val is not None and OPS[c["op"]](val, c["value"])
            res.update(observed=f"{c['metric']}={val} (need {c['op']} {c['value']})", passed=ok)
        results.append(res)
    passed = bool(results) and all(x["passed"] for x in results)
    out = {"recipe": r["id"], "run_dir": str(run_dir), "mode": mode, "pass": passed, "checks": results}
    (run_dir / "verify.json").write_text(json.dumps(out, indent=2))
    if passed:
        progress = load_progress()
        progress[r["id"]] = {"mode": mode, "run": str(run_dir), "at": datetime.now().isoformat(timespec="seconds")}
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    lines = [f"verify {r['id']} ({mode}): {'PASS' if passed else 'FAIL'}"]
    for x in results:
        lines.append(f"  {'PASS' if x['passed'] else 'FAIL'}  {x['id']:<24} {x['observed']}")
        if not x["passed"] and x["hint"]:
            lines.append(f"        hint: {x['hint']}")
    if not metrics and not any("file" in c for c in r.get("checks", [])):
        lines.append("  (no metrics.json found - did the run finish?)")
    lines.append("")
    lines.append("Next: rleng inspect   # look at the actual outputs, not just the numbers"
                 if passed else "Next: rleng inspect   # find out why, fix, then `rleng run` again")
    emit(args, out, "\n".join(lines))
    return 0 if passed else 1


def cmd_verify(args) -> int:
    return verify_run(args, resolve_run(args.run))


def cmd_inspect(args) -> int:
    run_dir = resolve_run(args.run)
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    rows_path = run_dir / "rollouts.jsonl"
    rows = [json.loads(l) for l in rows_path.open()] if rows_path.exists() else []
    by_kind: dict[str, list[float]] = {}
    for row in rows:
        if "kind" in row and "reward" in row:
            by_kind.setdefault(row["kind"], []).append(row["reward"])
    kinds = {k: {"n": len(v), "mean_reward": round(sum(v) / len(v), 3)} for k, v in sorted(by_kind.items())}
    scored = sorted((x for x in rows if "reward" in x), key=lambda x: x["reward"])
    n = args.n
    data = {"run_dir": str(run_dir), "metrics": metrics, "n_rollouts": len(rows), "by_kind": kinds,
            "lowest": scored[:n], "highest": scored[-n:][::-1]}

    def fmt(x):
        out = str(x.get("output", ""))[:160].replace("\n", "\\n")
        tag = f"[{x['kind']}] " if "kind" in x else (f"[step {x['step']}] " if "step" in x else "")
        prompt = f"\n         prompt: {str(x['prompt'])[:100]}" if isinstance(x.get("prompt"), str) else ""
        return f"  {x['reward']:+.3f} {tag}{out}{prompt}"

    lines = [f"run: {run_dir}", "", "metrics:"] + [f"  {k}: {v}" for k, v in metrics.items()]
    if kinds:
        lines += ["", "mean reward by kind:"] + [f"  {k:<20} n={v['n']:<4} {v['mean_reward']:+.3f}" for k, v in kinds.items()]
    if scored:
        lines += ["", f"highest reward ({n}):"] + [fmt(x) for x in data["highest"]]
        lines += ["", f"lowest reward ({n}):"] + [fmt(x) for x in data["lowest"]]
        lines += ["", f"all rollouts: {rows_path}"]
    emit(args, data, "\n".join(lines))
    return 0


def cmd_report(args) -> int:
    run_dir = resolve_run(args.run)
    v_path = run_dir / "verify.json"
    if not v_path.exists():
        raise UsageError("run `rleng verify` before reporting")
    v = json.loads(v_path.read_text())
    s = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    payload = {
        "recipe": v["recipe"], "recipe_version": s.get("recipe_version"), "rleng": __version__,
        "mode": v["mode"], "pass": v["pass"], "checks": {c["id"]: c["passed"] for c in v["checks"]},
        "platform": f"{sys.platform}-{platform.machine()}", "gpu": s.get("gpu"), "duration_s": s.get("duration_s"),
        "agent": args.agent,
    }
    title = f"[report] {payload['recipe']} {payload['mode']} {'PASS' if payload['pass'] else 'FAIL'} on {(payload['gpu'] or {}).get('name') or payload['platform']}"
    body = "Run report from `rleng report` (no code, prompts, or paths included):\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n\nNotes (optional):\n"
    url = f"https://github.com/{REPO}/issues/new?" + urllib.parse.urlencode({"title": title, "body": body, "labels": "run-report"})
    emit(args, {"payload": payload, "url": url},
         f"This is everything that would be shared:\n\n{json.dumps(payload, indent=2)}\n\nOpen this link to submit it (nothing is sent automatically):\n{url}")
    if args.open:
        webbrowser.open(url)
    return 0


# ---------- entry ----------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rleng", description="RL for engineers: verified, agent-runnable recipes.")
    p.add_argument("--version", action="version", version=f"rleng {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        sp.set_defaults(fn=fn)
        return sp

    add("list", cmd_list, "show the learning path")
    add("doctor", cmd_doctor, "check this machine and what it can run")
    sp = add("get", cmd_get, "copy a recipe into the current directory")
    sp.add_argument("id")
    sp.add_argument("--dir")
    sp.add_argument("--force", action="store_true")
    sp = add("run", cmd_run, "run a recipe's steps, then verify")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--mode", choices=["smoke", "full"], default="full")
    sp = add("verify", cmd_verify, "check a run against the recipe's pass criteria")
    sp.add_argument("run", nargs="?")
    sp = add("inspect", cmd_inspect, "show metrics and best/worst rollouts of a run")
    sp.add_argument("run", nargs="?")
    sp.add_argument("-n", type=int, default=5)
    sp = add("report", cmd_report, "prepare a shareable run report (opt-in, nothing sent automatically)")
    sp.add_argument("run", nargs="?")
    sp.add_argument("--agent", help="which coding agent ran this, if any (e.g. claude-code)")
    sp.add_argument("--open", action="store_true", help="open the report link in a browser")

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except UsageError as e:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(e)}))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
