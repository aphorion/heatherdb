#!/usr/bin/env python3
"""results/results.json + prices.json -> results/report.html (the shareable page)."""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / "results/results.json").read_text())
P = json.loads((HERE / "prices.json").read_text())
FINDINGS = "https://claude.ai/artifact/WKZfPrym9pcBibk3snMjQW"

SPEC = [  # task, primary metric, label, capability, incumbent role, what it replaces
    ("search", "recall@10", "recall@10", "Vector search", "vector database"),
    ("discovery", "nmi", "NMI", "Pattern discovery", "clustering job"),
    ("recs", "ndcg@10", "NDCG@10", "Recommendations", "recommender model"),
    ("fraud", "pr_auc", "PR-AUC", "Fraud detection", "outlier model"),
    ("classify", "roc_auc", "ROC-AUC", "Propensity scoring", "supervised model"),
]
DATASET = {
    "search": "Fashion-MNIST, 60,000 x 784, 1,000 queries",
    "discovery": "Fashion-MNIST, 60,000 train / 10,000 test, labels hidden",
    "recs": "MovieLens 1M, leave-last-out, 2,000 users",
    "fraud": "Credit-card fraud (ULB), 142,404 test transactions, 246 frauds",
    "classify": "UCI bank marketing, 31,647 train / 13,564 test",
}
e = html.escape


def instance_for(mem):
    need = mem * P["memory_headroom"] / 2**30
    return next((i for i in P["instances"] if i["mem_gib"] >= need), P["instances"][-1])


def f3(x):
    return f"{x:.3f}"


rows, bars, stack = [], [], []
for task, m, lab, cap, role in SPEC:
    sy = R["tasks"][task]["systems"]
    h = sy[0]
    inc = [s for s in sy[1:] if "mem_bytes" in s and "oracle" not in s["system"]
           and not s["system"].startswith("HeatherDB")]
    best = max(inc, key=lambda s: s[m])
    ratio = h[m] / best[m]
    v, cls = (("Parity or better", "good") if ratio >= 0.98 else
              ("Close", "warn") if ratio >= 0.90 else ("Gap", "bad"))
    rows.append(f"<tr><td><b>{e(cap)}</b><span class='sub'>{e(DATASET[task])}</span></td>"
                f"<td>{lab}</td><td class='num'>{f3(h[m])}</td>"
                f"<td class='num'>{f3(best[m])}<span class='sub'>{e(best['system'])}</span></td>"
                f"<td class='num'>{ratio:.2f}</td><td><span class='chip {cls}'>{v}</span></td></tr>")
    bars.append((cap, ratio, cls))
    stack.append((cap, role, best))

# Ratio chart: horizontal bars, x from 0 to 1.2, reference line at 1.0.
W, ROWH, LEFT, RIGHT, TOP = 640, 34, 150, 40, 26
plotw = W - LEFT - RIGHT
X = lambda r: LEFT + plotw * min(r, 1.2) / 1.2
svg = [f"<svg viewBox='0 0 {W} {TOP + ROWH * len(bars) + 30}' role='img' "
       f"aria-label='HeatherDB score as a share of the best incumbent, per capability'>"]
for t in (0, 0.25, 0.5, 0.75, 1.0):
    x = X(t)
    svg.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{TOP - 6}' y2='{TOP + ROWH * len(bars)}' "
               f"class='{'ref' if t == 1.0 else 'grid'}'/>")
    svg.append(f"<text x='{x:.1f}' y='{TOP + ROWH * len(bars) + 18}' class='tick' "
               f"text-anchor='middle'>{t:.2f}</text>")
svg.append(f"<text x='{X(1.0):.1f}' y='{TOP - 10}' class='tick' text-anchor='middle'>"
           f"= best incumbent</text>")
for i, (cap, r, cls) in enumerate(bars):
    y = TOP + i * ROWH + 7
    svg.append(f"<text x='{LEFT - 10}' y='{y + 14}' class='lbl' text-anchor='end'>{e(cap)}</text>")
    svg.append(f"<rect x='{LEFT}' y='{y}' width='{X(r) - LEFT:.1f}' height='20' rx='2' class='bar {cls}'/>")
    svg.append(f"<text x='{X(r) + 6:.1f}' y='{y + 14}' class='val'>{r:.2f}</text>")
svg.append("</svg>")

# Infrastructure
inf_rows, total = [], 0.0
for cap, role, best in stack:
    inst = instance_for(best["mem_bytes"])
    disk = best.get("disk_bytes", 0) / 2**30
    retrain = 0.0
    if best.get("process") == "python service" and "fit_s" in best:
        retrain = best["fit_s"] / 3600 * P["retrain_runs_per_month"] * inst["usd_per_hour"]
    cost = inst["usd_per_hour"] * P["hours_per_month"] + disk * P["block_storage_usd_per_gib_month"] + retrain
    total += cost
    inf_rows.append(f"<tr><td>{e(best['system'])}</td><td>{e(role)}</td>"
                    f"<td class='num'>{best['mem_bytes'] / 2**20:,.0f} MiB</td>"
                    f"<td>{inst['name']}</td><td class='num'>${cost:,.2f}</td></tr>")
fp = R["heather_footprint"]
hi = instance_for(fp["peak_rss_bytes"])
hcost = hi["usd_per_hour"] * P["hours_per_month"] + fp["disk_bytes"] / 2**30 * P["block_storage_usd_per_gib_month"]
n = len(stack)

fresh = R["tasks"]["recs"]["freshness"]
als = next(s for s in R["tasks"]["recs"]["systems"] if "retrain_to_add_new_user_s" in s)
fresh_x = als["retrain_to_add_new_user_s"] * 1000 / fresh["new_user_ready_ms_p50"]
disc = R["tasks"]["discovery"]["systems"]
k10 = disc[0]["at_k10"]
oracle = next(s for s in disc if "oracle" in s["system"])
search = R["tasks"]["search"]["systems"]


def sysrow(task, prefix):
    return next((x for x in R["tasks"][task]["systems"] if x["system"].startswith(prefix)), None)


def best_inc(task, m):
    return max((x for x in R["tasks"][task]["systems"][1:]
                if "mem_bytes" in x and "oracle" not in x["system"]
                and not x["system"].startswith("HeatherDB")), key=lambda x: x[m])


# Before/after selective attention, where the suite measured both.
sa_rows = []
for task, m, lab, cap in [("classify", "roc_auc", "ROC-AUC", "Propensity scoring"),
                          ("recs", "ndcg@10", "NDCG@10", "Recommendations")]:
    before, after = sysrow(task, "HeatherDB (no selective"), sysrow(task, "HeatherDB (selective")
    if before and after:
        b = best_inc(task, m)
        sa_rows.append(f"<tr><td><b>{cap}</b></td><td>{lab}</td><td class='num'>{f3(before[m])}</td>"
                       f"<td class='num'><b>{f3(after[m])}</b></td>"
                       f"<td class='num'>{f3(b[m])}<span class='sub'>{e(b['system'])}</span></td></tr>")
recs_sa = sysrow("recs", "HeatherDB (selective")
curve = recs_sa.get("learned_time_weights") if recs_sa else None
fraud_h = R["tasks"]["fraud"]["systems"][0]
fraud_b = best_inc("fraud", "pr_auc")
good = sum(1 for _, r, _ in bars if r >= 0.98)
close = sum(1 for _, r, _ in bars if 0.90 <= r < 0.98)
gap = len(bars) - good - close
_parts = [f"matched or beat the incumbent on {good}"] if good else []
if close:
    _parts.append(f"came within 10% on {close}")
if gap:
    _parts.append(f"fell short on {gap}")
outcome_sentence = (", ".join(_parts[:-1]) + " and " + _parts[-1]) if len(_parts) > 1 else _parts[0]


def detail(task, keys):
    sy = R["tasks"][task]["systems"]
    head = "".join(f"<th>{e(lbl)}</th>" for _, lbl, _ in keys)
    body = []
    for s in sy:
        cells = "".join(
            f"<td class='num'>{fmt(s[k])}</td>" if k in s else "<td class='num dim'>-</td>"
            for k, _, fmt in keys)
        body.append(f"<tr><td>{e(s['system'])}</td>{cells}</tr>")
    return (f"<div class='scroll'><table><thead><tr><th>System</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


ms = lambda x: f"{x:,.1f} ms"
us = lambda x: f"{x:,.0f} µs"
sec = lambda x: f"{x:,.1f} s"
num = lambda x: f"{x:,.0f}"

page = f"""<title>HeatherDB Head-to-Head</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: a single reading column of lab-report sections; tables scroll on their own. */
:root {{
  --bg: #fbfafc; --fg: #1d1a22; --muted: #625b6c; --line: #e3dfe8; --panel: #f3f0f6;
  --accent: #6d3f86;
  --good: #1f7a4d; --good-bg: #e2f3ea; --warn: #8a5a00; --warn-bg: #fbefd6; --bad: #a3312b; --bad-bg: #f8e1df;
  --sans: "Instrument Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #16131a; --fg: #ece8f1; --muted: #a49cae; --line: #2f2a36; --panel: #201c25; --accent: #c39bd9;
  --good: #6fd3a0; --good-bg: #173527; --warn: #f0c060; --warn-bg: #3a2d10; --bad: #f08c84; --bad-bg: #3d1c1a;
  color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #16131a; --fg: #ece8f1; --muted: #a49cae; --line: #2f2a36; --panel: #201c25; --accent: #c39bd9;
  --good: #6fd3a0; --good-bg: #173527; --warn: #f0c060; --warn-bg: #3a2d10; --bad: #f08c84; --bad-bg: #3d1c1a;
  color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--sans); }}
main {{ max-width: 920px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 40px; }}
section {{ display: grid; gap: 14px; min-width: 0; }}
.eyebrow {{ font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font-size: clamp(28px, 5vw, 40px); line-height: 1.1; margin: 0; text-wrap: balance; letter-spacing: -.01em; }}
h2 {{ font-size: 20px; margin: 0; text-wrap: balance; }}
h3 {{ font-size: 15px; margin: 8px 0 0; }}
p {{ margin: 0; max-width: 68ch; }}
.lead {{ font-size: 17px; color: var(--fg); }}
.muted, .sub {{ color: var(--muted); }}
.sub {{ display: block; font-size: 12.5px; font-weight: 400; }}
.figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.fig {{ border-top: 2px solid var(--accent); padding-top: 10px; display: grid; gap: 2px; }}
.fig b {{ font: 600 28px/1.1 var(--mono); }}
.fig span {{ font-size: 13px; color: var(--muted); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font: 600 12px/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }}
td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.dim {{ color: var(--muted); }}
tr.total td {{ font-weight: 700; border-bottom: 0; }}
.chip {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12.5px; font-weight: 600; white-space: nowrap; }}
.chip.good {{ color: var(--good); background: var(--good-bg); }}
.chip.warn {{ color: var(--warn); background: var(--warn-bg); }}
.chip.bad {{ color: var(--bad); background: var(--bad-bg); }}
svg {{ width: 100%; height: auto; max-width: 720px; }}
svg .grid {{ stroke: var(--line); }} svg .ref {{ stroke: var(--fg); stroke-dasharray: 3 3; }}
svg .tick {{ fill: var(--muted); font: 11px var(--mono); }} svg .lbl {{ fill: var(--fg); font: 13px var(--sans); }}
svg .val {{ fill: var(--fg); font: 600 12px var(--mono); }}
svg .bar.good {{ fill: var(--good); }} svg .bar.warn {{ fill: var(--warn); }} svg .bar.bad {{ fill: var(--bad); }}
ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 72ch; }}
pre {{ background: var(--panel); padding: 14px 16px; border-radius: 6px; overflow-x: auto; font: 13px/1.5 var(--mono); margin: 0; }}
a {{ color: var(--accent); }}
code {{ font: 13px var(--mono); }}
</style>
<main>
<header style="display:grid;gap:12px">
  <div class="eyebrow">Aphorion Labs · reproducible benchmark · {R['finished'][:10]}</div>
  <h1>HeatherDB Head-to-Head</h1>
  <p class="lead">One HeatherDB process ran {n} enterprise workloads, each against the specialist
  system it would replace, on the same public data, splits and metrics. It {outcome_sentence}. Running all {n} costs
  {total / hcost:.1f}x less per month than the specialist stack: one system to operate
  instead of {n + 1}.</p>
</header>

<div class="figures">
  <div class="fig"><b>{total / hcost:.1f}x</b><span>lower monthly infrastructure (${total:,.0f} vs ${hcost:,.0f})</span></div>
  <div class="fig"><b>{n + 1} → 1</b><span>systems to operate, including the retraining scheduler</span></div>
  <div class="fig"><b>{fresh_x:,.0f}x</b><span>faster from new data to serving ({fresh['new_user_ready_ms_p50']:.0f} ms vs a {als['retrain_to_add_new_user_s']:.1f} s ALS refit)</span></div>
  <div class="fig"><b>{good + close} of {n}</b><span>workloads at parity or within 10% of the best incumbent</span></div>
</div>

<section>
  <h2>Accuracy on the same task</h2>
  <p class="muted">HeatherDB's score as a share of the best incumbent's on each workload's primary metric. 1.00 means a tie.</p>
  {''.join(svg)}
  <div class="scroll"><table>
    <thead><tr><th>Workload</th><th>Metric</th><th>HeatherDB</th><th>Best incumbent</th><th>Ratio</th><th>Verdict</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
</section>

<section>
  <h2>Infrastructure for all {n} workloads</h2>
  <p class="muted">Each incumbent is deployed as its own service on the smallest instance that holds its measured
  peak memory with {P['memory_headroom']}x headroom; models that must refit to take in new data add
  {P['retrain_runs_per_month']} retrain runs a month. HeatherDB's peak memory is measured with all {n} databases
  loaded in one process. Prices: AWS us-east-1 on-demand, Graviton m7g, list prices at time of writing.</p>
  <div class="scroll"><table>
    <thead><tr><th>Component</th><th>Replaces</th><th>Peak memory</th><th>Instance</th><th>USD / month</th></tr></thead>
    <tbody>{''.join(inf_rows)}
    <tr class="total"><td>Specialist stack</td><td>{n} systems + a retraining scheduler</td><td></td><td></td><td class="num">${total:,.2f}</td></tr>
    <tr class="total"><td>HeatherDB</td><td>1 process, {n} databases</td><td class="num">{fp['peak_rss_bytes'] / 2**20:,.0f} MiB</td><td>{hi['name']}</td><td class="num">${hcost:,.2f}</td></tr>
    </tbody></table></div>
</section>

<section>
  <h2>What selective attention changed</h2>
  <p>Selective attention lets the memory learn which features and which moments to weigh, from prediction error,
  while data is written: each row is predicted before it is stored, and the miss teaches the attention. Same
  engine, same data; the only difference is the database setting.</p>
  <div class="scroll"><table>
    <thead><tr><th>Workload</th><th>Metric</th><th>Without</th><th>With</th><th>Best incumbent</th></tr></thead>
    <tbody>{''.join(sa_rows)}</tbody></table></div>
  {f"<p class='muted'>Learned recency, relative to the newest item (lags 1, 2, 3-4, 5-7, 8-12, …): {', '.join(f'{w:.2f}' for w in curve[:8])}. It started flat; nobody set a decay.</p>" if curve else ""}
</section>

<section>
  <h2>Remaining gaps</h2>
  <ul>
    <li><b>Fraud precision.</b> PR-AUC {fraud_h['pr_auc']:.3f} against {fraud_b['system']}'s {fraud_b['pr_auc']:.3f}
    (ROC-AUC is level or better). The read scores raw distance to normal; LOF also corrects for local density.
    The fix, a spread per memory location, reached PR-AUC 0.500 in the prototype and is the next engine change.</li>
    <li><b>Recommendations.</b> NDCG@10 {recs_sa['ndcg@10']:.3f} against {best_inc('recs', 'ndcg@10')['system']}'s
    {best_inc('recs', 'ndcg@10')['ndcg@10']:.3f} ({recs_sa['ndcg@10'] / best_inc('recs', 'ndcg@10')['ndcg@10']:.0%}). Learned recency closed
    most of the gap; what is left is the item code, a fixed random projection where ALS learns item vectors. With
    ALS's item vectors the same read reached 0.093 in the prototype, so codes the memory learns itself are the next lever.</li>
    <li><b>Cost per scored row.</b> Search is now faster than Qdrant ({search[0]['p50_ms']:.1f} ms against
    {search[1]['p50_ms']:.1f} ms p50), but per-row scoring still costs more than a fitted model's: a propensity recall
    {sysrow('classify', 'HeatherDB (selective')['score_us_per_row'] / 1000:.1f} ms against
    {best_inc('classify', 'roc_auc')['score_us_per_row']:.1f} µs, because the read scans every stored row, exactly, on
    every request. Fine for interactive use; batch scoring of millions of rows needs an index. The cost model sizes by
    memory, so at high query volume HeatherDB needs more CPU than shown.</li>
  </ul>
  <p>Diagnoses and proposed fixes: <a href="{FINDINGS}" target="_blank" rel="noopener">engine findings log</a>.</p>
</section>

<section>
  <h2>Workload detail</h2>
  <h3>Vector search · {e(DATASET['search'])}</h3>
  {detail('search', [('recall@10', 'recall@10', f3), ('build_s', 'build', sec), ('p50_ms', 'p50', ms), ('p95_ms', 'p95', ms)])}
  <h3>Pattern discovery · {e(DATASET['discovery'])}</h3>
  {detail('discovery', [('k', 'K', num), ('nmi', 'NMI', f3), ('ari', 'ARI', f3), ('purity', 'purity', f3), ('fit_s', 'fit', sec)])}
  <p class="muted">HeatherDB is not given K: minimum description length chose K={disc[0]['k']} from its learned codebook
  (this selection currently runs client-side; finding F8). Cut at the true K=10, the same codebook scores
  NMI {k10['nmi']:.3f} (k-means given K=10: {oracle['nmi']:.3f}). The chosen K varies between runs (80 and 115 in the
  last two) because competitive writes are not yet seeded; NMI held at 0.523 in both.</p>
  <h3>Recommendations · {e(DATASET['recs'])}</h3>
  {detail('recs', [('hr@10', 'HR@10', f3), ('ndcg@10', 'NDCG@10', f3), ('fit_s', 'fit', sec)])}
  <h3>Fraud detection · {e(DATASET['fraud'])}</h3>
  {detail('fraud', [('roc_auc', 'ROC-AUC', f3), ('pr_auc', 'PR-AUC', f3), ('recall@1%fpr', 'recall @ 1% FPR', f3), ('score_us_per_txn', 'per txn', us)])}
  <h3>Propensity scoring · {e(DATASET['classify'])}</h3>
  {detail('classify', [('roc_auc', 'ROC-AUC', f3), ('pr_auc', 'PR-AUC', f3), ('fit_s', 'fit', sec), ('score_us_per_row', 'per row', us)])}
</section>

<section>
  <h2>Method</h2>
  <ul>
    <li>Public datasets, downloaded from their canonical sources; a SHA-256 fingerprint of the exact arrays is recorded per run.</li>
    <li>HeatherDB is used only through its public HTTP API, with authentication on. Incumbents use library defaults or standard settings.</li>
    <li>Qdrant runs as the real server in Docker. The other incumbents run as separate Python processes, each measured alone.</li>
    <li>One machine: {e(R['machine']['platform'])}, Apple M4, 16 GiB. Engine <code>{e(R['heather']['version'])}</code> at <code>{R['heather']['commit']} ({R['heather'].get('branch', '')})</code>.</li>
    <li>Not modelled: staff time to run each system, network between services, high availability and replicas.</li>
  </ul>
  <pre>cargo build --release -p heather_server
cd benchmarks/h2h
uv venv .venv --python 3.12 &amp;&amp; uv pip install --python .venv/bin/python -r requirements.txt
docker pull qdrant/qdrant:v1.12.4
.venv/bin/python run.py &amp;&amp; .venv/bin/python report.py &amp;&amp; .venv/bin/python render_html.py</pre>
</section>
</main>
"""
(HERE / "results/report.html").write_text(page)
print("wrote", HERE / "results/report.html")
