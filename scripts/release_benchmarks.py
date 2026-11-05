#!/usr/bin/env python3
"""Capture native Criterion and local authenticated HTTP benchmark evidence."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
import release_acceptance as fixture

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--collect-only', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
fixture.binary = args.binary.resolve()
target = Path(os.environ.get('CARGO_TARGET_DIR', root / 'target'))
suites = [('read_latency', 'read/dimension'), ('write_throughput', 'write/single'), ('write_throughput', 'write/batch/(individual|batch)/100$')]
commands = []
for suite, pattern in suites:
    command = ['cargo', 'bench', '--locked', '-p', 'heather_db', '--bench', suite, '--', pattern, '--sample-size', '20', '--warm-up-time', '1', '--measurement-time', '2']
    commands.append(command)
    if not args.collect_only:
        subprocess.run(command, cwd=root, check=True)

results = []
cases = [(f'{group}/d/{d}', kind, d) for group, kind in [('read_dimension', 'read'), ('write_single', 'write')] for d in [64,128,384]]
cases += [(f'write_batch/{mode}/100', mode, 128) for mode in ['individual', 'batch']]
for case, kind, dimension in cases:
    directory = target / 'criterion' / case / 'new'
    estimates = json.loads((directory/'estimates.json').read_text())
    samples = json.loads((directory/'sample.json').read_text())
    assert len(samples['times']) == 20
    results.append({'id': case, 'kind': kind, 'dimension': dimension, 'measuredAt': datetime.datetime.fromtimestamp((directory/'sample.json').stat().st_mtime, datetime.timezone.utc).isoformat(), 'meanNs': estimates['mean']['point_estimate'], 'mean95CiNs': [estimates['mean']['confidence_interval']['lower_bound'], estimates['mean']['confidence_interval']['upper_bound']], 'estimates': estimates, 'samples': samples})

http = []
with tempfile.TemporaryDirectory(prefix='heather-http-bench-') as directory:
    engine = fixture.Engine(directory)
    try:
        engine.start()
        engine.create('bench')
        route = '/db/bench/collections/memory'
        vectors = [fixture.unit(i) for i in range(32)]
        assert engine.request('POST', route+'/write', {'vectors':vectors})[0] == 200
        status, payload = engine.request('POST', '/auth/token')
        assert status == 200
        for auth_name, authorization in [('Basic', engine.auth), ('Bearer', 'Bearer '+payload['token'])]:
            for _ in range(10):
                assert engine.request('POST', route+'/read', {'query':fixture.unit(0)}, auth=authorization)[0] == 200
            samples_ms = []
            for _ in range(100):
                start = time.perf_counter_ns()
                assert engine.request('POST', route+'/read', {'query':fixture.unit(0)}, auth=authorization)[0] == 200
                samples_ms.append((time.perf_counter_ns()-start)/1e6)
            ordered = sorted(samples_ms)
            http.append({'operation':'POST read', 'auth':auth_name, 'dimension':64, 'competitive':False, 'precision':'f64', 'sync':True, 'corpusVectors':32, 'clients':1, 'warmupRequests':10, 'requests':100, 'meanMs':statistics.mean(samples_ms), 'p50Ms':statistics.median(samples_ms), 'p95Ms':ordered[math.ceil(.95*len(ordered))-1], 'samplesMs':samples_ms})
    finally:
        engine.stop()

def command(*args):
    return subprocess.check_output(args, cwd=root, text=True).strip()

files = ['Cargo.lock','heather_db/src/config.rs','heather_db/src/read.rs','heather_db/src/collection.rs','heather_db/benches/helpers.rs','heather_db/benches/read_latency.rs','heather_db/benches/write_throughput.rs']
report = {'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'engineCommit':command('git','rev-parse','HEAD'), 'engineVersion':'1.0.0', 'targetRelease':'1.0.0', 'releaseStatus':'Local 1.0.0 candidate; publication requires the release audit gates', 'hardware':{'cpu':command('sysctl','-n','machdep.cpu.brand_string'), 'memoryBytes':int(command('sysctl','-n','hw.memsize')), 'os':command('sw_vers','-productVersion'), 'architecture':platform.machine()}, 'rustc':command('rustc','--version'), 'nativeProtocol':{'samples':20, 'warmupSeconds':1, 'targetMeasurementSeconds':2, 'mode':'competitive=true, default beta=5, data-seeded l0=0, f64, synchronous LMDB', 'input':'unseeded normalized normal vectors', 'readCorpusVectors':200, 'readQueries':100, 'measurement':'Criterion sample mean; 95% CI is not request p95', 'batch':'128 dimensions, 100 vectors, 200 prefill writes outside timing; timed closure includes teardown'}, 'httpProtocol':'Sequential loopback urllib requests; new connection per request; includes client JSON serialization/parsing, transport and authentication; no concurrent-load throughput claim; Basic measured before Bearer', 'sourceHashes':{file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in files}, 'commands':commands, 'native':results, 'http':http}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'native':[{k:v for k,v in x.items() if k in ['id','meanNs','mean95CiNs']} for x in results], 'http':[{k:v for k,v in x.items() if k!='samplesMs'} for x in http]},indent=2))
