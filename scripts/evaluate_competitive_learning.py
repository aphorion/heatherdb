#!/usr/bin/env python3
"""Frozen synthetic learning/discovery evaluation; actual HeatherDB writes."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
import scipy
from scipy import stats
import sklearn
from sklearn.cluster import KMeans, HDBSCAN
import release_acceptance as fixture

D, GROUPS, TEST_N = 128, 8, 128
SEEDS = [11, 29, 47, 71, 101]
CHECKPOINTS = [64, 256, 1024]
BETAS = [5, 50]


def unit(rows):
    rows = np.asarray(rows, dtype=float)
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-30)


def cosine_rows(predictions, targets):
    return np.sum(unit(predictions) * unit(targets), axis=1)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observations(prototypes, n, noise, rng, imbalance=False, background=0):
    groups = len(prototypes)
    if imbalance:
        labels = rng.choice(groups, n, p=[.65] + [.05] * 7)
    else:
        labels = np.concatenate([rng.permutation(np.tile(np.arange(groups), int(np.ceil(64/groups))))[:64] for _ in range(int(np.ceil(n/64)))])[:n]
        # Evaluation panels are exactly class-balanced, regardless of K.
        if n == TEST_N and groups == GROUPS:
            labels = rng.permutation(np.repeat(np.arange(groups), n//groups))
    noise_vectors = unit(rng.normal(size=(n, D)))
    values = unit(prototypes[labels] + noise * noise_vectors)
    if background:
        is_noise = rng.random(n) < background
        values[is_noise] = unit(rng.normal(size=(int(is_noise.sum()), D)))
        labels[is_noise] = -1
    return values, labels


def model(keys, values, counts):
    return {'keys': np.asarray(keys), 'values': np.asarray(values), 'counts': np.asarray(counts, dtype=float)}


def common_decode(memory, queries, beta=50, k=20):
    dots = queries @ memory['keys'].T
    take = min(k, dots.shape[1])
    indices = np.argsort(-dots, axis=1, kind='stable')[:, :take]
    selected = np.take_along_axis(dots, indices, axis=1)
    logits = beta * selected + np.log(np.maximum(memory['counts'][indices], 1e-30))
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum('nk,nkd->nd', weights, memory['values'][indices])


def online_centroids(values, budget, alpha=None):
    budget = min(len(values), max(1, budget))
    means = values[:budget].copy()
    keys = unit(means)
    counts = np.ones(budget)
    for value in values[budget:]:
        index = int(np.argmax(keys @ value))
        counts[index] += 1
        means[index] += (alpha if alpha is not None else 1/counts[index]) * (value-means[index])
        keys[index] = unit(means[index])
    return model(keys, means, counts)


def labelled_reference(values, labels):
    kinds = sorted(set(int(x) for x in labels if x >= 0))
    means = np.array([values[labels == c].mean(axis=0) for c in kinds])
    counts = np.array([(labels == c).sum() for c in kinds])
    return model(unit(means), means, counts)


def baselines(values, labels, budgets, seed):
    memories = {'plain_all': model(values, values, np.ones(len(values)))}
    average = values.mean(axis=0, keepdims=True)
    memories['global_mean'] = model(unit(average), average, [len(values)])
    permutation = np.random.default_rng(seed + 40000).permutation(len(values))
    for tag, budget in budgets.items():
        sampled = values[permutation[:min(len(values), budget)]]
        memories['reservoir_' + tag] = model(sampled, sampled, np.ones(len(sampled)))
        memories['online_kmeans_' + tag] = online_centroids(values, budget)
    sampled = values[permutation[:min(len(values), 2*budgets['default'])]]
    memories['reservoir_double_default'] = model(sampled, sampled, np.ones(len(sampled)))
    memories['online_kmeans_8'] = online_centroids(values, 8)
    memories['online_ema_8'] = online_centroids(values, 8, .05)
    km = KMeans(n_clusters=8, n_init=5, max_iter=100, random_state=seed).fit(values)
    means = km.cluster_centers_
    memories['offline_kmeans_8'] = model(unit(means), means, np.bincount(km.labels_, minlength=8))
    memories['labelled_reference'] = labelled_reference(values, labels)
    return memories


def score(memory, queries, labels, prototypes, old_panel=None):
    targets = prototypes[labels]
    nearest = np.argmax(queries @ memory['keys'].T, axis=1)
    outputs = {f'attention_{beta}': common_decode(memory, queries, beta) for beta in BETAS}
    outputs['nearest_address'] = memory['keys'][nearest]
    outputs['nearest_value'] = memory['values'][nearest]
    arrays = {key: cosine_rows(value, targets) for key, value in outputs.items()}
    metrics = {key + '_cosine': float(value.mean()) for key, value in arrays.items()}
    metrics['per_class_cosine'] = [float(arrays['attention_50'][labels == c].mean()) for c in range(len(prototypes))]
    metrics['diagnostic_class_accuracy'] = float(np.mean(np.argmax(unit(outputs['attention_50']) @ prototypes.T, axis=1) == labels))
    metrics['address_prototype_coverage'] = float(np.max(prototypes @ unit(memory['keys']).T, axis=1).mean())
    metrics['value_prototype_coverage'] = float(np.max(prototypes @ unit(memory['values']).T, axis=1).mean())
    metrics['locations'] = len(memory['keys'])
    metrics['logical_float_payload_bytes'] = int(memory['keys'].nbytes + memory['counts'].nbytes + (0 if memory['keys'] is memory['values'] else memory['values'].nbytes))
    if old_panel is not None:
        old_x, old_y, old_p = old_panel
        metrics['old_pattern_cosine'] = float(cosine_rows(common_decode(memory, old_x), old_p[old_y]).mean())
    return metrics, arrays


class ActualMemory:
    def __init__(self, engine, auth, name, competitive, k):
        self.engine, self.auth, self.name, self.k = engine, auth, name, k
        status, payload = engine.request('POST', '/db', {'name':name, 'dimension':D, 'map_size_mb':64, 'eam':{'competitive':competitive, 'k':k, 'precision':'f64'}}, auth=auth)
        if status not in (200,201):
            raise RuntimeError(f'Database creation failed: {status}: {payload}')
        self.route = f'/db/{name}/collections/learning'

    def write(self, values):
        for offset in range(0, len(values), 128):
            chunk = values[offset:offset+128]
            status, payload = self.engine.request('POST', self.route+'/write', {'vectors':chunk.tolist()}, auth=self.auth)
            if status != 200 or payload['count'] != len(chunk):
                raise RuntimeError(f'Write failed: {status}: {payload}')

    def snapshot(self):
        status, payload = self.engine.request('GET', self.route+'/locations?full=true', auth=self.auth)
        if status != 200:
            raise RuntimeError(f'Snapshot failed: {status}')
        rows = payload['locations']
        counts = np.array([r['write_count'] for r in rows])
        return model(np.array([r['address'] for r in rows]), np.array([r['counter'] for r in rows]) / np.maximum(counts[:, None], 1e-30), counts)

    def spot_check(self, memory, queries, targets):
        attention, iterative = [], []
        for query in queries[:8]:
            status, payload = self.engine.request('POST', self.route+'/attention', {'query':query.tolist(), 'scale':50}, auth=self.auth)
            if status != 200:
                raise RuntimeError('Native attention check failed')
            attention.append(payload['result'])
            status, payload = self.engine.request('POST', self.route+'/read', {'query':query.tolist(), 'strategy':'iterative'}, auth=self.auth)
            if status != 200:
                raise RuntimeError('Native iterative check failed')
            iterative.append(payload['result'])
        exact = common_decode(memory, queries[:8], 50, self.k)
        return {'queries':8, 'configured_k':self.k, 'native_attention_cosine':float(cosine_rows(attention,targets[:8]).mean()), 'native_iterative_beta5_cosine':float(cosine_rows(iterative,targets[:8]).mean()), 'exact_configured_k_attention_cosine':float(cosine_rows(exact,targets[:8]).mean()), 'native_vs_exact_max_component_difference':float(np.max(np.abs(np.array(attention)-exact)))}


def proposals(memory, use_addresses=False):
    directions = unit(memory['keys'] if use_addresses else memory['values'])
    priority = np.linalg.norm(memory['values'],axis=1)*np.sqrt(memory['counts'])
    order = np.argsort(-priority,kind='stable')
    accepted = []
    for index in order:
        if memory['counts'][index] < 4:
            continue
        direction = directions[index]
        if not accepted or np.max(np.array(accepted) @ direction) < .95:
            accepted.append(direction)
        if len(accepted) == 64:
            break
    return np.array(accepted).reshape(-1,D)


def validate_discovery(directions, validation, prototypes):
    count = len(directions)
    if not count:
        return {'proposed':0, 'validated':0, 'matched_validated':0, 'hidden_groups_covered':0, 'matched_validated_fraction':None, 'validation_hits':[], 'p_values':[]}
    hits = np.sum(validation @ directions.T > .5, axis=0)
    null_cap = stats.beta.sf(.75, (D-1)/2, (D-1)/2)
    pvalues = np.maximum(stats.binom.sf(hits-1,len(validation),null_cap),1e-300)
    passes = (hits >= 8) & (pvalues < .05/count)
    selected = directions[passes]
    if len(prototypes) and len(selected):
        matches = selected @ prototypes.T >= .9
        covered = int(np.any(matches,axis=0).sum())
        matched = int(np.any(matches,axis=1).sum())
    else:
        covered, matched = 0, 0
    return {'proposed':count, 'validated':int(passes.sum()), 'matched_validated':matched, 'hidden_groups_covered':covered, 'matched_validated_fraction':float(matched/passes.sum()) if passes.sum() else None, 'validation_hits':hits.tolist(), 'p_values':pvalues.tolist(), 'isotropic_cap_probability':float(null_cap)}


def interval(values, confidence=.95):
    values = np.asarray(values)
    mean = float(values.mean())
    if len(values) < 2:
        return {'mean':mean,'interval':None,'seeds':len(values)}
    half = float(stats.t.ppf((1+confidence)/2,len(values)-1)*stats.sem(values))
    return {'mean':mean,'interval':[mean-half,mean+half],'seeds':len(values),'confidence':confidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture.binary = args.binary.resolve()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    checkpoints = CHECKPOINTS[:2] if args.smoke else CHECKPOINTS
    args.artifacts.mkdir(parents=True,exist_ok=True)
    rows, discoveries, artifacts = [], [], []
    started = time.perf_counter()

    def save(name, payload):
        file = args.artifacts / (name+'.npz')
        np.savez_compressed(file,**payload)
        artifacts.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size})

    with tempfile.TemporaryDirectory(prefix='heather-learning-') as directory:
        engine = fixture.Engine(directory)
        try:
            engine.start()
            status, token = engine.request('POST','/auth/token')
            assert status == 200
            auth = 'Bearer '+token['token']
            for seed in seeds:
                qr,_ = np.linalg.qr(np.random.default_rng(seed).normal(size=(D,16)))
                initial = qr[:,:8].T
                changed = (initial+qr[:,8:16].T)/np.sqrt(2)
                conditions = [('balanced',.75,False),('hard_noise',1.5,False),('imbalanced',.75,True),('drift',.75,False)]
                for ci,(condition,noise,imbalance) in enumerate(conditions):
                    rng = np.random.default_rng(seed*1000+ci)
                    if condition == 'drift':
                        first,first_y = observations(initial,256,noise,rng)
                        new,new_y = observations(changed,max(checkpoints),noise,rng)
                        train,labels = np.vstack([first,new]),np.concatenate([first_y,new_y])
                        steps = [256]+[256+n for n in checkpoints]
                        current_p = changed
                    else:
                        train,labels = observations(initial,max(checkpoints),noise,rng,imbalance)
                        steps,current_p = checkpoints,initial
                    test_x,test_y = observations(current_p,TEST_N,noise,np.random.default_rng(seed*1000+ci+200000))
                    old_x,old_y = observations(initial,TEST_N,noise,np.random.default_rng(seed*1000+ci+300000))
                    actual = {tag:ActualMemory(engine,auth,f's{seed}_{ci}_{tag}',True,k) for tag,k in [('default',20),('winner',1)]}
                    plain = ActualMemory(engine,auth,f's{seed}_{ci}_plain',False,20)
                    previous = 0
                    for n in steps:
                        for memory in [*actual.values(),plain]:
                            memory.write(train[previous:n])
                        previous = n
                        learned = {tag:memory.snapshot() for tag,memory in actual.items()}
                        plain_memory = plain.snapshot()
                        # Plain storage must have one verbatim key/value per write.
                        np.testing.assert_allclose(plain_memory['keys'],train[:n],atol=1e-14,rtol=1e-14)
                        np.testing.assert_allclose(plain_memory['values'],train[:n],atol=1e-14,rtol=1e-14)
                        budgets = {tag:len(value['keys']) for tag,value in learned.items()}
                        memories = baselines(train[:n],labels[:n],budgets,seed)
                        memories.update({'competitive_'+tag:memory for tag,memory in learned.items()})
                        target = current_p[test_y]
                        raw = {'train':train[:n], 'train_labels':labels[:n], 'test':test_x,'test_labels':test_y,'prototypes':current_p}
                        measurements = {}
                        for name,memory in memories.items():
                            metrics,arrays = score(memory,test_x,test_y,current_p,(old_x,old_y,initial) if condition=='drift' else None)
                            measurements[name] = metrics
                            for key,value in arrays.items():raw[name+'_'+key] = value
                        measurements['input'] = {'attention_50_cosine':float(cosine_rows(test_x,target).mean()),'attention_5_cosine':float(cosine_rows(test_x,target).mean()),'locations':0}
                        native = {tag:memory.spot_check(learned[tag],test_x,target) for tag,memory in actual.items()}
                        for tag,memory in learned.items():
                            for key,value in memory.items():raw[tag+'_'+key] = value
                        record = {'seed':seed,'condition':condition,'noise':noise,'writes':n,'post_drift_writes':n-256 if condition=='drift' else None,'data_hash':digest(train[:n]),'test_hash':digest(test_x),'measurements':measurements,'native_spot_checks':native}
                        rows.append(record)
                        save(f'{condition}_s{seed}_n{n}',raw)
                        print(json.dumps({'condition':condition,'seed':seed,'writes':n,'locations':budgets,'default_cosine':measurements['competitive_default']['attention_50_cosine'],'input_cosine':measurements['input']['attention_50_cosine']}),flush=True)
                    for memory in [*actual.values(),plain]:engine.request('DELETE','/db/'+memory.name,auth=auth)

                if args.smoke:
                    discovery_cases = [5,0]
                else:
                    discovery_cases = [5,9,0]
                for groups in discovery_cases:
                    rng = np.random.default_rng(seed*1000+500+groups)
                    p = qr[:,:groups].T
                    n = 256 if args.smoke else 1024
                    if groups:
                        x,y = observations(p,n,.75,rng,background=.3)
                        validation,_ = observations(p,n,.75,rng,background=.3)
                    else:
                        x,validation = unit(rng.normal(size=(n,D))),unit(rng.normal(size=(n,D)))
                        y = np.full(n,-1)
                    actual = {tag:ActualMemory(engine,auth,f'ds{seed}_g{groups}_{tag}',True,k) for tag,k in [('default',20),('winner',1)]}
                    learned = {}
                    for tag,memory in actual.items():
                        memory.write(x)
                        learned[tag] = memory.snapshot()
                    candidates = {}
                    average = x.mean(axis=0,keepdims=True)
                    candidates['global_mean'] = proposals(model(unit(average),average,[len(x)]))
                    for tag,memory in learned.items():
                        candidates['competitive_'+tag] = proposals(memory)
                        candidates['addresses_'+tag] = proposals(memory,True)
                        budget = min(len(x),len(memory['keys']))
                        subset = x[np.random.default_rng(seed+40000).permutation(len(x))[:budget]]
                        indices = np.argsort(-(subset @ subset.T),axis=1,kind='stable')[:,:min(20,budget)]
                        averages = subset[indices].mean(axis=1)
                        candidates['local_average_'+tag] = proposals(model(subset,averages,np.full(budget,min(20,budget))))
                    clusters = HDBSCAN(min_cluster_size=16,min_samples=8).fit_predict(x)
                    if np.any(clusters>=0):
                        candidates['hdbscan'] = proposals(labelled_reference(x,clusters))
                    else:candidates['hdbscan'] = np.empty((0,D))
                    candidates['labelled_reference'] = proposals(labelled_reference(x,y)) if groups else np.empty((0,D))
                    results = {name:validate_discovery(directions,validation,p) for name,directions in candidates.items()}
                    discoveries.append({'seed':seed,'hidden_groups':groups,'train':n,'validation':n,'data_hash':digest(x),'methods':results})
                    payload = {'train':x,'labels':y,'validation':validation,'prototypes':p}
                    payload.update(candidates)
                    save(f'discovery_s{seed}_g{groups}',payload)
                    short = lambda value:{key:value[key] for key in ['proposed','validated','hidden_groups_covered']}
                    print(json.dumps({'discovery_seed':seed,'hidden_groups':groups,'default':short(results['competitive_default']),'winner':short(results['competitive_winner'])}),flush=True)
                    for memory in actual.values():engine.request('DELETE','/db/'+memory.name,auth=auth)
        finally:engine.stop()

    comparisons = []
    for condition in ['balanced','hard_noise','imbalanced','drift']:
        selected = [r for r in rows if r['condition']==condition and r['writes']==max(x['writes'] for x in rows if x['condition']==condition)]
        for method in ['competitive_default','competitive_winner']:
            tag = 'default' if method.endswith('default') else 'winner'
            for baseline in ['input','plain_all','reservoir_'+tag,'online_kmeans_'+tag,'offline_kmeans_8','reservoir_double_default']:
                differences = [r['measurements'][method]['attention_50_cosine']-r['measurements'][baseline]['attention_50_cosine'] for r in selected]
                primary = condition=='balanced' and method=='competitive_default' and baseline in ['input','reservoir_default']
                comparisons.append({'condition':condition,'method':method,'baseline':baseline,'primary':primary,'paired_difference':interval(differences,.975 if primary else .95)})
    source = ['heather_db/src/write.rs','heather_db/src/collection.rs','heather_db/src/config.rs','scripts/evaluate_competitive_learning.py','docs/reference/competitive-learning-protocol.md']
    git = lambda *argv:subprocess.check_output(['git',*argv],cwd=root,text=True).strip()
    report = {'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'smoke_only':args.smoke,'engineCommit':git('rev-parse','HEAD'),'binaryVersion':subprocess.check_output([str(fixture.binary),'--version'],text=True).strip(),'binarySha256':hashlib.sha256(fixture.binary.read_bytes()).hexdigest(),'sourceHashes':{file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in source},'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'architecture':platform.machine()},'protocol':{'dimension':D,'groups':GROUPS,'seeds':seeds,'checkpoints':checkpoints,'test_examples':TEST_N,'primary_readout':'exact common top20, count-prior attention scale50; same for all representations','engine_split_rng':'not exposed; actual snapshots retained','production_algorithms_modified':False,'logical_payload':'keys+values+counts; aliased key/value arrays counted once; not RSS or disk size','discovery_validation':'independent spherical-cap/binomial test under known isotropic synthetic null; minimum8 hits above cosine0.5, Bonferroni0.05/proposals','not_a_new_scientific_discovery':True},'learning':rows,'comparisons':comparisons,'discovery':discoveries,'artifacts':artifacts,'elapsedSeconds':time.perf_counter()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {len(rows)} learning checkpoints and {len(discoveries)} discovery runs.',flush=True)


if __name__=='__main__':main()
