#!/usr/bin/env python3
"""results/results.json + prices.json -> results/REPORT.md (accuracy + cost)."""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / "results/results.json").read_text())
P = json.loads((HERE / "prices.json").read_text())

# task -> (primary metric, secondary metric, capability, incumbent role)
SPEC = {
    "search":    ("recall@10", "p50_ms", "Vector search", "vector database"),
    "discovery": ("nmi", "purity", "Pattern discovery / segmentation", "clustering job"),
    "recs":      ("ndcg@10", "hr@10", "Recommendations", "recommender model"),
    "fraud":     ("pr_auc", "roc_auc", "Fraud / anomaly detection", "outlier model"),
    "classify":  ("roc_auc", "pr_auc", "Propensity classification", "supervised model"),
}
LOWER_BETTER = {"p50_ms"}


def instance_for(mem_bytes):
    need = mem_bytes * P["memory_headroom"] / 2**30
    for inst in P["instances"]:
        if inst["mem_gib"] >= need:
            return inst
    return P["instances"][-1]


def monthly(inst):
    return inst["usd_per_hour"] * P["hours_per_month"]


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


lines = ["# Head-to-head: one HeatherDB vs the specialist stack", "",
         f"Run {R['started']} to {R['finished']} on {R['machine']['platform']}. "
         f"Engine `{R['heather']['version']}` at `{R['heather']['commit']} ({R['heather'].get('branch', '')})`.", ""]

# ---------------------------------------------------------------- accuracy
lines += ["## Accuracy, same data, same task", "",
          "| Capability | Primary metric | HeatherDB | Best incumbent | Ratio | Verdict |",
          "|---|---|---|---|---|---|"]
stack = []
for task, (m1, m2, cap, role) in SPEC.items():
    if task not in R["tasks"]:
        continue
    systems = R["tasks"][task]["systems"]
    heather = systems[0]
    inc = [s for s in systems[1:] if "mem_bytes" in s and "oracle" not in s["system"]
           and not s["system"].startswith("HeatherDB")]
    best = max(inc, key=lambda s: s.get(m1, -1))
    ratio = heather[m1] / best[m1] if best[m1] else float("nan")
    verdict = ("parity or better" if ratio >= 0.98 else
               "close (within 10 %)" if ratio >= 0.90 else "gap")
    lines.append(f"| {cap} | {m1} | {fmt(heather[m1])} | {fmt(best[m1])} ({best['system']}) "
                 f"| {ratio:.2f} | {verdict} |")
    stack.append((cap, role, best))

for task, (m1, m2, cap, role) in SPEC.items():
    if task not in R["tasks"]:
        continue
    lines += ["", f"### {cap}", "", "| System | " + " | ".join(
        k for k in R["tasks"][task]["systems"][0] if isinstance(R["tasks"][task]["systems"][0][k], (int, float)))
        + " |"]
    keys = [k for k, v in R["tasks"][task]["systems"][0].items() if isinstance(v, (int, float))]
    lines.append("|---" * (len(keys) + 1) + "|")
    for s in R["tasks"][task]["systems"]:
        lines.append(f"| {s['system']} | " + " | ".join(fmt(s.get(k)) for k in keys) + " |")
    if "at_k10" in R["tasks"][task]["systems"][0]:
        a = R["tasks"][task]["systems"][0]["at_k10"]
        lines += ["", f"At the true K=10 (an oracle choice), HeatherDB's codebook scores NMI "
                      f"{a['nmi']:.3f} / ARI {a['ari']:.3f} / purity {a['purity']:.3f}."]
    if "freshness" in R["tasks"][task]:
        f = R["tasks"][task]["freshness"]
        als = next((s for s in R["tasks"][task]["systems"] if "retrain_to_add_new_user_s" in s), None)
        lines += ["", f"Freshness: a new user's history is servable in "
                      f"{f['new_user_ready_ms_p50']:.0f} ms (p50) / {f['new_user_ready_ms_p95']:.0f} ms (p95) "
                      f"on HeatherDB" + (f", against a {als['retrain_to_add_new_user_s']:.1f} s ALS refit "
                                          f"({als['retrain_to_add_new_user_s'] * 1000 / f['new_user_ready_ms_p50']:.0f}x)."
                                          if als else ".")]

# -------------------------------------------------------------------- cost
fp = R["heather_footprint"]
h_inst = instance_for(fp["peak_rss_bytes"])
h_disk = fp["disk_bytes"] / 2**30
h_cost = monthly(h_inst) + h_disk * P["block_storage_usd_per_gib_month"]

lines += ["", "## Infrastructure: everything above, deployed", "",
          f"Each incumbent is deployed as its own service on the smallest instance holding its "
          f"measured peak memory x{P['memory_headroom']}. Models that must be refit to take new "
          f"data add {P['retrain_runs_per_month']} retrain runs a month. Prices: {P['_note']}", "",
          "| Component | Role | Peak memory | Disk | Instance | USD / month |",
          "|---|---|---|---|---|---|"]
total, n_sys = 0.0, 0
for cap, role, best in stack:
    inst = instance_for(best["mem_bytes"])
    disk = best.get("disk_bytes", 0) / 2**30
    retrain = 0.0
    if "fit_s" in best and best.get("process") == "python service" and cap != "Vector search":
        retrain = best["fit_s"] / 3600 * P["retrain_runs_per_month"] * inst["usd_per_hour"]
    cost = monthly(inst) + disk * P["block_storage_usd_per_gib_month"] + retrain
    total += cost
    n_sys += 1
    lines.append(f"| {best['system']} | {role} | {best['mem_bytes'] / 2**20:.0f} MiB | {disk:.2f} GiB "
                 f"| {inst['name']} | {cost:.2f} |")
lines += [f"| **Specialist stack** | {n_sys} systems + a retraining scheduler | | | | **{total:.2f}** |",
          f"| **HeatherDB** | 1 process, {len(R['tasks'])} databases | {fp['peak_rss_bytes'] / 2**20:.0f} MiB "
          f"| {h_disk:.2f} GiB | {h_inst['name']} | **{h_cost:.2f}** |", "",
          f"HeatherDB serves all {len(R['tasks'])} capabilities from one process; its peak memory is "
          f"measured with every database loaded at once. Ratio: {total / h_cost:.1f}x lower monthly "
          f"infrastructure, {n_sys + 1} systems to 1."]

(HERE / "results/REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
