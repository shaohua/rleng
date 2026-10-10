"""Every CPU recipe must FAIL as shipped (it's an exercise) and PASS with the reference solution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOLUTIONS = Path(__file__).parent / "solutions"
CPU_RECIPES = {"eval-an-agent": "verifier.py", "design-a-reward": "reward.py"}


def rleng(*args, cwd):
    env = {**os.environ, "RLENG_HOME": str(cwd / ".rleng-home")}
    return subprocess.run([sys.executable, "-m", "rleng.cli", *args, "--json"], cwd=cwd, capture_output=True, text=True, env=env)


@pytest.mark.parametrize("recipe,edit_file", CPU_RECIPES.items())
def test_starting_file_fails_and_solution_passes(tmp_path, recipe, edit_file):
    assert rleng("get", recipe, cwd=tmp_path).returncode == 0
    work = tmp_path / recipe

    before = rleng("run", cwd=work)
    assert before.returncode == 1, before.stderr[-2000:]
    assert json.loads(before.stdout)["pass"] is False

    shutil.copy(SOLUTIONS / f"{recipe}__{edit_file.replace('.py', '')}.py", work / edit_file)
    after = rleng("run", cwd=work)
    assert after.returncode == 0, after.stdout + after.stderr[-2000:]
    assert json.loads(after.stdout)["pass"] is True

    insp = rleng("inspect", cwd=work)
    assert insp.returncode == 0 and json.loads(insp.stdout)["n_rollouts"] > 0
    rep = rleng("report", cwd=work)
    assert rep.returncode == 0 and json.loads(rep.stdout)["payload"]["pass"] is True


def test_list_and_doctor(tmp_path):
    ids = [r["id"] for r in json.loads(rleng("list", cwd=tmp_path).stdout)["recipes"]]
    assert ids[:2] == ["eval-an-agent", "design-a-reward"]
    assert "gpu" in json.loads(rleng("doctor", cwd=tmp_path).stdout)


def test_unknown_recipe_is_usage_error(tmp_path):
    r = rleng("get", "nope", cwd=tmp_path)
    assert r.returncode == 3 and "unknown recipe" in json.loads(r.stdout)["error"]
