"""Build site/index.html from recipes/*/recipe.yaml and harness/results/*.jsonl."""

import html
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
INSTALL = "uv tool install git+https://github.com/shaohua/rleng"
recipes = sorted((yaml.safe_load(p.read_text()) for p in (ROOT / "src/rleng/recipes").glob("*/recipe.yaml")),
                 key=lambda r: r["order"])

edits = {r["id"]: set(r.get("edit", [])) for r in recipes}
agent_stats: dict[str, list[bool]] = {}
for f in (ROOT / "harness/results").glob("*.jsonl"):
    for line in f.read_text().splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            agent_stats.setdefault(row["recipe"], []).append(row["pass"] == "True" and {c for c in row["changed"] if not c.startswith("+")} <= edits.get(row["recipe"], set()))  # "+x" = new file (logs), not an edit


def card(r):
    hw = r.get("hardware", {})
    need = f"GPU · {hw.get('min_vram_gb', '?')} GB+" if hw.get("gpu") else "CPU only"
    verified = ", ".join(hw.get("verified", [])) if hw.get("gpu") else "any laptop"
    runs = agent_stats.get(r["id"], [])
    agent = f"{sum(runs)}/{len(runs)} unaided agent runs pass" if runs else "agent pass rate: not measured yet"
    prompt = (f"Take me through the rleng recipe '{r['id']}'. If `rleng` isn't installed: `{INSTALL}`. "
              f"Run `rleng get {r['id']}`, read its AGENT.md first, and follow it.")
    return f"""
    <article class="card">
      <div class="num">{r['order']}</div>
      <div class="body">
        <h3>{html.escape(r['title'])}</h3>
        <p>{html.escape(r['summary'])}</p>
        <ul class="meta"><li>{need}</li><li>~{r.get('time', {}).get('full', '?')}</li><li>verified: {html.escape(verified or 'pending')}</li><li>{agent}</li></ul>
        <div class="actions">
          <button data-copy="{html.escape(prompt)}">Copy for agent</button>
          <code>rleng get {r['id']}</code>
        </div>
      </div>
    </article>"""


page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>rleng: RL for engineers who ship agents</title>
<meta name="description" content="Short, verified, agent-runnable recipes: agent evals, reward design, and your first GRPO run on a small GPU.">
<style>
:root{{--ink:#16181d;--muted:#5d6370;--line:#e4e6ea;--accent:#0b6e4f;--bg:#fbfbf9}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif;color:var(--ink);background:var(--bg)}}
main{{max-width:760px;margin:0 auto;padding:64px 20px}}
h1{{font-size:2.2rem;line-height:1.15;margin:0 0 12px;letter-spacing:-.02em}}
.lede{{font-size:1.15rem;color:var(--muted);margin:0 0 28px}}
pre,code{{font:14px ui-monospace,SFMono-Regular,Menlo,monospace}}
pre{{background:#16181d;color:#e8e8e3;padding:16px 18px;border-radius:8px;overflow-x:auto}}
h2{{margin:48px 0 16px;font-size:1.1rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.card{{display:flex;gap:18px;border:1px solid var(--line);background:#fff;border-radius:10px;padding:20px;margin:0 0 14px}}
.num{{font-weight:700;font-size:1.4rem;color:var(--accent);min-width:1.4em}}
.card h3{{margin:0 0 4px;font-size:1.1rem}}.card p{{margin:0 0 10px;color:var(--muted)}}
.meta{{list-style:none;padding:0;margin:0 0 14px;display:flex;flex-wrap:wrap;gap:6px}}
.meta li{{font-size:.8rem;border:1px solid var(--line);border-radius:99px;padding:2px 10px;color:var(--muted)}}
.actions{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
button{{font:inherit;font-size:.9rem;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:7px 14px;cursor:pointer}}
button.done{{background:var(--ink)}}
.why li{{margin-bottom:6px}}footer{{margin-top:56px;color:var(--muted);font-size:.9rem}}a{{color:var(--accent)}}
</style></head><body><main>
<h1>RL for engineers who ship agents.</h1>
<p class="lede">Short recipes that take you from evaluating an agent to training a small model with GRPO against a
reward you designed. Every recipe has a pass/fail your coding agent can check.</p>
<pre>{INSTALL}
rleng list
rleng get eval-an-agent &amp;&amp; cd eval-an-agent
rleng run       # PASS / FAIL with hints
rleng inspect   # read the actual outputs</pre>
<h2>The path</h2>
{''.join(card(r) for r in recipes)}
<h2>Why this exists</h2>
<ul class="why">
<li><b>Verified, not aspirational.</b> <code>rleng verify</code> checks real metrics. Hardware badges come from real runs.</li>
<li><b>Agent-first.</b> Stable CLI, JSON output, an <code>AGENT.md</code> per recipe, and a measured pass rate for unaided coding agents.</li>
<li><b>Inspect, don't just plot.</b> Every recipe ends with reading real outputs.</li>
</ul>
<footer>For theory: <a href="https://walkinglabs.github.io/hands-on-modern-rl/">hands-on-modern-rl</a>.
For fast fine-tuning notebooks: <a href="https://unsloth.ai/docs">Unsloth</a>. rleng is the practical bridge.
· <a href="https://github.com/shaohua/rleng">GitHub</a></footer>
</main>
<script>
document.querySelectorAll('button[data-copy]').forEach(b=>b.onclick=async()=>{{
  await navigator.clipboard.writeText(b.dataset.copy);b.textContent='Copied';b.classList.add('done');
  setTimeout(()=>{{b.textContent='Copy for agent';b.classList.remove('done')}},1500)}});
</script></body></html>
"""
(ROOT / "site").mkdir(exist_ok=True)
(ROOT / "site/index.html").write_text(page)
print(f"site/index.html: {len(recipes)} recipes, agent stats for {sorted(agent_stats)}")
