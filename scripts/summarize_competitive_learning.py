#!/usr/bin/env python3
"""Render the frozen learning evaluation into tables, curves and a report."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('results',type=Path)
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
r = json.loads(args.results.read_text())
assert not r['smoke_only']
assert len(r['learning']) == 65 and len(r['discovery']) == 15
figures = args.output.parent/'figures'
figures.mkdir(parents=True,exist_ok=True)

def cases(condition, final=True):
    rows = [x for x in r['learning'] if x['condition']==condition]
    return [x for x in rows if x['writes']==max(y['writes'] for y in rows)] if final else rows


def mean(condition,method,key='attention_50_cosine'):
    return float(np.mean([x['measurements'][method][key] for x in cases(condition)]))


def ci(values,confidence=.95):
    values = np.array(values)
    m = float(values.mean())
    half = float(stats.t.ppf((1+confidence)/2,len(values)-1)*stats.sem(values))
    return m,m-half,m+half


def change(condition,key):
    rows = cases(condition,False)
    steps = sorted({x['writes'] for x in rows})
    gains = []
    for seed in r['protocol']['seeds']:
        first = next(x for x in rows if x['seed']==seed and x['writes']==steps[0])
        last = next(x for x in rows if x['seed']==seed and x['writes']==steps[-1])
        gains.append(last['measurements']['competitive_default'][key]-first['measurements']['competitive_default'][key])
    return ci(gains)


methods = ['input','competitive_default','competitive_winner','plain_all','reservoir_default','reservoir_double_default','online_kmeans_default','online_kmeans_8','online_ema_8','offline_kmeans_8','labelled_reference']
labels = {'input':'Noisy input','competitive_default':'Competitive default','competitive_winner':'Competitive k=1','plain_all':'Plain: all observations','reservoir_default':'Reservoir: same count','reservoir_double_default':'Reservoir: twice count','online_kmeans_default':'Online k-means: same count','online_kmeans_8':'Online k-means: K=8','online_ema_8':'Online EMA: K=8','offline_kmeans_8':'Offline k-means: K=8','labelled_reference':'Labelled centroid reference'}
conditions = [('balanced','Balanced, noise 0.75'),('hard_noise','Balanced, noise 1.5'),('imbalanced','Imbalanced exposure'),('drift','After 45° pattern drift')]

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
colors = {'competitive_default':'#266b8a','plain_all':'#7c7287','reservoir_default':'#ac8044','online_kmeans_default':'#518160','input':'#999999'}
fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,(condition,title) in zip(axes.flat,conditions):
    rows = cases(condition,False)
    steps = sorted({x['writes'] for x in rows})
    for method in colors:
        means,lo,hi=[],[],[]
        for n in steps:
            values=[x['measurements'][method]['attention_50_cosine'] for x in rows if x['writes']==n]
            m,l,h=ci(values);means.append(m);lo.append(l);hi.append(h)
        x=np.array(steps)-(256 if condition=='drift' else 0)
        ax.plot(x,means,marker='o',markersize=4,label=labels[method],color=colors[method],linewidth=1.7)
        ax.fill_between(x,lo,hi,color=colors[method],alpha=.12)
    ax.set_title(title,loc='left',fontweight='bold')
    ax.set_xlabel('New observations after drift' if condition=='drift' else 'Training observations')
    ax.set_ylabel('Held-out clean-prototype cosine')
    ax.set_ylim(.45,1.015)
    ax.grid(alpha=.15)
handles,legend=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,legend,loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Competitive learning: new examples, shared latent patterns\nFive data seeds; identical exact top-20 readout, scale 50; bands are 95% seed-level t intervals',fontsize=12)
fig.savefig(figures/'competitive-learning.png',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
dm=['competitive_default','competitive_winner','local_average_default','hdbscan']
dl=['Default','k=1','Local mean','HDBSCAN']
dc=['#266b8a','#6898a8','#ac8044','#518160']
for group,ax in zip([5,9],axes[:2]):
    rows=[x for x in r['discovery'] if x['hidden_groups']==group]
    values=[np.mean([x['methods'][m]['hidden_groups_covered']/group for x in rows]) for m in dm]
    purity=[np.mean([x['methods'][m]['matched_validated_fraction'] or 0 for x in rows]) for m in dm]
    x=np.arange(4)
    ax.bar(x-.18,values,.34,color=dc,alpha=.95)
    ax.bar(x+.18,purity,.34,color=dc,alpha=.35,hatch='//')
    ax.set_xticks(x,dl,rotation=25,ha='right')
    ax.set_title(f'{group} hidden groups',loc='left',fontweight='bold')
    ax.set_ylim(0,1.12);ax.set_ylabel('Fraction');ax.grid(axis='y',alpha=.15)
null=[x for x in r['discovery'] if x['hidden_groups']==0]
proposed=[sum(x['methods'][m]['proposed'] for x in null) for m in dm]
validated=[sum(x['methods'][m]['validated'] for x in null) for m in dm]
axes[2].bar(np.arange(4),proposed,color=dc,alpha=.3,label='Proposed')
axes[2].scatter(np.arange(4),validated,color='#111111',marker='x',s=35,zorder=3,label='Validated')
axes[2].set_xticks(np.arange(4),dl,rotation=25,ha='right')
axes[2].set_title('Five structure-free streams',loc='left',fontweight='bold')
axes[2].set_ylabel('Total candidate directions');axes[2].legend(frameon=False,fontsize=8)
axes[2].grid(axis='y',alpha=.15)
fig.suptitle('Discovery calibration: coverage is strong; proposal parsimony differs\nIndependent validation rejects null proposals; this rejection is supplied by the evaluation layer',fontsize=12)
fig.legend(handles=[Patch(facecolor='#777777',label='Hidden-group coverage'),Patch(facecolor='#dddddd',edgecolor='#555555',hatch='//',label='Matched / validated')],loc='outside lower center',ncol=2,frameon=False)
fig.savefig(figures/'competitive-discovery.png',dpi=170)
plt.close(fig)

table=[]
for method in methods:
    table.append('| '+labels[method]+' | '+' | '.join(f'{mean(c,method):.4f}' for c,_ in conditions)+' |')
primary=[]
for item in r['comparisons']:
    if item['primary']:
        value=item['paired_difference'];low,high=value['interval']
        primary.append(f'| Default − {labels[item["baseline"]]} | {value["mean"]:+.4f} | [{low:+.4f}, {high:+.4f}] |')
discovery_table=[]
for group in [5,9]:
    rows=[x for x in r['discovery'] if x['hidden_groups']==group]
    for name in ['competitive_default','competitive_winner','local_average_default','hdbscan','global_mean']:
        p=np.mean([x['methods'][name]['proposed'] for x in rows]);v=np.mean([x['methods'][name]['validated'] for x in rows]);coverage=np.mean([x['methods'][name]['hidden_groups_covered'] for x in rows]);purity=np.mean([x['methods'][name]['matched_validated_fraction'] or 0 for x in rows])
        discovery_table.append(f'| {group} | {name} | {p:.1f} | {v:.1f} | {coverage:.1f}/{group} | {purity:.1%} |')
growth=[]
for c,_ in conditions:
    m,l,h=change(c,'attention_50_cosine')
    growth.append(f'| {c} | {m:+.4f} | [{l:+.4f}, {h:+.4f}] |')
default=mean('balanced','competitive_default')
locations=mean('balanced','competitive_default','locations')
balanced_curve={n:float(np.mean([x['measurements']['competitive_default']['attention_50_cosine'] for x in cases('balanced',False) if x['writes']==n])) for n in r['protocol']['checkpoints']}
default_purities=[float(np.mean([x['methods']['competitive_default']['matched_validated_fraction'] for x in r['discovery'] if x['hidden_groups']==groups])) for groups in [5,9]]
endpoints=[]
for c,_ in conditions:
    z=cases(c)
    endpoints.append('| '+c+' | '+' | '.join(f'{np.mean([x["native_spot_checks"]["default"][key] for x in z]):.4f}' for key in ['native_attention_cosine','native_iterative_beta5_cosine'])+' |')
old_first=[x for x in cases('drift',False) if x['writes']==256]
old_last=cases('drift')
old_before=np.mean([x['measurements']['competitive_default']['old_pattern_cosine'] for x in old_first])
old_after=np.mean([x['measurements']['competitive_default']['old_pattern_cosine'] for x in old_last])

report=f'''# Competitive learning and discovery evaluation

5 November 2026. Tested repository revision `{r['engineCommit']}`;
binary `{r['binaryVersion']}`. Engine production code is unchanged from the
audited 1.0.0 candidate. The [protocol](../reference/competitive-learning-protocol.md)
was frozen before the full run; a one-seed smoke run checked the pipeline.

## Finding

**Competitive mode learns shared patterns in this synthetic evaluation.**
It improves prediction of clean latent structure from new noisy observations,
improves as observations accumulate, and adapts after patterns change. This
is a different capability from preserving and recalling every individual row.

On the primary balanced task, default competitive writes reached mean held-out
prototype cosine **{default:.4f}** after 1,024 writes, versus
**{mean('balanced','input'):.4f}** for the noisy input,
**{mean('balanced','reservoir_default'):.4f}** for a same-count reservoir and
**{mean('balanced','reservoir_double_default'):.4f}** for a twice-count reservoir.
The codebook had **{locations:.1f} locations on average** (1,024/{locations:.1f}
= {1024/locations:.2f}× fewer locations than observations). A location contains
both address and counter information: this is not a {1024/locations:.2f}× RAM or
disk compression claim.

**Competitive mode is also a viable candidate generator for hidden structure
on these calibration tasks.** Every planted group was covered across all five
seeds, for both five and nine groups. Default mode generates many mixed or
redundant candidates; HDBSCAN is markedly more parsimonious here. The external
validation layer rejects null candidates. Native memory does not automatically
validate or abstain, and this is not a new scientific discovery.

## What was learned and how it was tested

The generator supplies noisy observations of orthogonal shared prototypes.
Clean prototypes are never written into the database. No labels or group counts
are passed to competitive writing. New test examples have independent noise.
Training prefixes and test panels are identical across methods, at 128 dimensions
and five data seeds. Each held-out panel has 128 examples, balanced by class.

Actual authenticated HeatherDB writes produce the exported addresses, counters
and counts. For a controlled comparison, all representations use an exact
top-20, count-weighted attention decoder. Primary results fix scale 50 for every
method; scale 5, nearest-address and nearest-counter diagnostics are retained
in the raw data. Baseline clustering is evaluator code, not part of HeatherDB.
Neither the engine nor its defaults was altered for these tests.

The main result is denoising toward a latent shared prototype, not matching a
stored observation. Counts, per-class scores, native-endpoint checks and raw
codebooks prevent interpreting a plausible output alone as learning.

## Held-out prototype fidelity

Mean scale-50 cosine at the final checkpoint, averaged over five data seeds.
The drift column is after 256 old and 1,024 new observations; other columns
use 1,024 observations. Imbalance is scored on a balanced test panel.

| Method | Balanced σ=0.75 | Hard noise σ=1.5 | Imbalanced | After drift |
|---|---|---|---|---|
{chr(10).join(table)}

K=8 baselines receive the known generator complexity. The labelled reference
additionally receives training labels and is not an unsupervised contender.
Count-matched baselines receive the competitive codebook count for that
training prefix. The reservoir is a uniform sample of the prefix, recomputed
at each checkpoint; its marginal distribution is reservoir sampling, not a
separately measured live reservoir service. The twice-count reservoir is a
logical two-vector-payload sensitivity check; graphs, caches, storage overhead
and exact byte-budget equivalence are not included.

The simpler baselines matter: count-matched online k-means slightly exceeds
default competitive mode on balanced, imbalanced and drift fidelity, while
competitive mode exceeds it in the high-noise condition. Offline K=8 k-means
is better on balanced and high-noise data. These results establish useful
learning, not universal superiority over clustering methods.

## Primary paired comparisons

The two primary comparisons use five paired seed-level means. Intervals are
97.5% t intervals for two-comparison Bonferroni coverage; queries are not
treated as independent experimental replicates. Both intervals exclude zero.

| Comparison in balanced final condition | Mean difference | 97.5% interval |
|---|---|---|
{chr(10).join(primary)}

The study has five seeds and synthetic data. These model-based intervals do
not establish performance over arbitrary real-world distributions. Other
comparisons and conditions are secondary.

## Learning curves and adaptation

Default competitive balanced fidelity grows from {balanced_curve[64]:.4f} at 64
writes to {balanced_curve[256]:.4f} at 256 and {default:.4f} at 1,024. More noisy observations
improve prediction of independent clean structure, rather than just adding
individually retrievable rows.

| Condition | Default final − first fidelity | Paired 95% interval |
|---|---|---|
{chr(10).join(growth)}

The drift gain starts before any new-pattern observations. Adaptation has a
retention cost: old-pattern fidelity falls from **{old_before:.4f}** to
**{old_after:.4f}**. This is updating the learned representation, not preserving
an immutable episodic archive. Archive requirements need a separate storage
policy; the experiment does not promise both unlimited retention and adaptation.

![Learning curves](figures/competitive-learning.png)

## Native readout cross-check

Eight preselected held-out queries per seed and checkpoint are also sent to
the actual HTTP attention and iterative-read endpoints. Final default-mode
means below are therefore over 40 native queries per condition, not the full
128-query panels used above.

| Condition | Native attention scale 50 | Native iterative read, default β=5 |
|---|---|---|
{chr(10).join(endpoints)}

Native attention agrees with the exact decoder at configured k to numerical
precision in these spot checks. These are quality checks, not latency or
throughput measurements. The k=1 variant uses a top-20 decoder in the controlled
comparison but has native top-1 reads; its controlled score must not be
presented as its native read score. All native k=1 results are retained.

## Discovery: proposals, validation and clean group recovery

Each stream contains either five or nine hidden groups plus 30% isotropic
background, or no groups at all. Training and validation each contain 1,024
independent observations. Methods do not receive the hidden group count.
Candidate directions are generated and deduplicated from training data before
validation, with a 64-proposal cap. Validation uses the frozen spherical-cap
null and binomial/Bonferroni rule in the protocol.

Means across five seeds. A matched candidate has cosine at least 0.9 to a
planted group prototype. Coverage counts each group once; the last column is
the mean per-seed fraction of validated candidates matching an individual
planted group.

| Hidden groups | Method | Proposed | Validated | Groups covered | Matched / validated |
|---|---|---|---|---|---|
{chr(10).join(discovery_table)}

Default competitive mode covers all groups, but only about {min(default_purities):.0%}–{max(default_purities):.0%} of its
validated candidates align closely with an individual planted prototype.
The remainder include mixed directions: validation detects enrichment relative
to isotropic noise, not necessarily a distinct interpretable group. They are
not automatically all false findings, but should not be advertised as distinct
discoveries. Winner-only learning yields much cleaner candidate directions
here, at a different native readout/denoising tradeoff. HDBSCAN finds one clean
validated centroid per hidden group, so this run does not show a discovery
advantage over that simple baseline.

On all **five structure-free streams**, default and winner-only modes still
produce proposals, but **zero are validated**. The local-average baseline and
HDBSCAN also have zero validated candidates on these nulls. This is a small
calibration set, not an empirical false-positive guarantee for arbitrary data.
The validation rule assumes the known isotropic synthetic null. Real data need
domain-appropriate nulls, confound controls and independent replication.

The labelled centroid reference passes the positive control. Thus these
directions are representable and detectable by this encoding and validator;
the null result is not simply an inability to express prototype structure.

![Discovery calibration](figures/competitive-discovery.png)

## Supported product framing and remaining work

- Supported: competitive writes learn a compressed representation of repeated
  shared structure and can improve prediction on unseen noisy examples.
- Supported here: the learned representation adapts to a change in shared
  patterns, with measurable loss of old-pattern fidelity.
- Supported as a calibration result: learned counters can propose all planted
  hidden groups, with independent validation rejecting the tested nulls.
- Not established: general semantic reasoning, a novel scientific discovery,
  superiority to standard clustering, or native automatic discovery validation.
- Discovery work remains in stable candidate selection, deduplication,
  interpretation and real-data validation. Do not change the default to k=1
  from this one study: its cleaner proposals do not dominate native denoising.

Individual-record recall and learning shared structure are different contracts.
The earlier random-pattern reconstruction result remains true; this experiment
provides the missing positive evidence for learning under repeated structure.

## Evidence and reproduction

- [Frozen protocol](../reference/competitive-learning-protocol.md)
- [All checkpoint, baseline, comparison and discovery results](competitive-learning-results.json)
- Source hashes, executable checksum, package versions, dataset hashes and
  snapshot checksums are included in that JSON. Actual codebooks and score
  arrays are in `output/competitive-learning-v1/` in the engine checkout;
  {len(r['artifacts'])} NPZ artifacts, approximately {sum(x['bytes'] for x in r['artifacts'])/1024**2:.1f} MiB, are retained locally
  rather than committed as generated binary data.
- Public API overload-split randomness has no exposed seed. Inputs are seeded,
  but this is one engine realization per data seed, not bitwise replay.
- A one-seed reduced smoke run tested implementation before the frozen full
  run. No production algorithm or hyperparameter changes followed full results.

Use Python 3.12 with `scripts/competitive-learning-requirements.txt` and
Matplotlib to render figures. From the engine checkout:

```bash
cargo build --release --locked -p heather_server
python scripts/evaluate_competitive_learning.py --binary target/release/heather --output /tmp/learning.json --artifacts /tmp/learning-artifacts
python scripts/summarize_competitive_learning.py /tmp/learning.json --output /tmp/competitive-learning.md
```

The binary must match the source being evaluated. All database data and
credentials are temporary; no production data or service is used.
'''
args.output.write_text(report)
with args.output.with_suffix('.csv').open('w') as file:
    writer=csv.writer(file);writer.writerow(['seed','condition','writes','method','cosine_scale50','cosine_scale5','locations','logical_float_payload_bytes'])
    for case in r['learning']:
        for method,value in case['measurements'].items():
            writer.writerow([case['seed'],case['condition'],case['writes'],method,value['attention_50_cosine'],value['attention_5_cosine'],value['locations'],value.get('logical_float_payload_bytes')])
print(args.output)
