#!/usr/bin/env python3
"""Black-box 1.0 acceptance checks. Uses only disposable local data directories."""
import argparse
import base64
import concurrent.futures
import datetime
import json
import math
import os
from pathlib import Path
import random
import platform
import secrets
import socket
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
binary = None
checks = []
observations = {}
timings = {}


def check(name, fn):
    started = time.perf_counter()
    try:
        detail = fn()
        checks.append({'name': name, 'status': 'pass', 'detail': detail})
    except Exception as exc:
        checks.append({'name': name, 'status': 'fail', 'detail': str(exc)[:500]})
    checks[-1]['seconds'] = round(time.perf_counter() - started, 3)
    print(f"{checks[-1]['status'].upper()}: {name}" + (f" — {checks[-1]['detail']}" if checks[-1]['status'] == 'fail' else ''), flush=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


class Engine:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.password = secrets.token_urlsafe(20)
        self.auth = 'Basic ' + base64.b64encode(f'admin:{self.password}'.encode()).decode()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.process = None
        self.stderr = None

    def start(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('HEATHER_')}
        env.update(HEATHER_ADMIN_USER='admin', HEATHER_ADMIN_PASSWORD=self.password, RUST_LOG='error')
        if self.stderr is not None:
            self.stderr.close()
        self.stderr = tempfile.TemporaryFile(mode='w+t')
        self.process = subprocess.Popen([str(binary), '--data-dir', str(self.directory), '--host', '127.0.0.1', '--port', str(self.port), '--dimension', '64', '--map-size-mb', '32'], env=env, stdout=subprocess.DEVNULL, stderr=self.stderr)
        for _ in range(200):
            if self.process.poll() is not None:
                self.stderr.seek(0)
                raise RuntimeError(f'Engine exited during startup: {self.process.returncode}: {self.stderr.read()[-1000:]}')
            try:
                if self.request('GET', '/ready', auth=False)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(.05)
        raise TimeoutError('Engine did not become ready in 10 seconds')

    def stop(self, kill=False):
        if self.process and self.process.poll() is None:
            self.process.kill() if kill else self.process.terminate()
            self.process.wait(timeout=15)

    def request(self, method, route, data=None, auth=None):
        headers = {'Content-Type': 'application/json'}
        if auth is not False:
            headers['Authorization'] = self.auth if auth is None else auth
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}{route}', body, headers, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=20)
        except urllib.error.HTTPError as exc:
            response = exc
        raw = response.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw.decode(errors='replace')[:300]
        return response.status, payload

    def cli(self, *cmd, directory=None):
        return subprocess.run([str(binary), '--data-dir', str(directory or self.directory), *cmd], capture_output=True, text=True, timeout=30)

    def create(self, name, competitive=False, map_size=16, precision='f64'):
        status, payload = self.request('POST', '/db', {'name': name, 'dimension': 64, 'map_size_mb': map_size, 'eam': {'competitive': competitive, 'precision': precision}})
        expect(status in (200, 201), f'Create database returned {status}: {payload}')


def unit(index):
    return [1.0 if i == index % 64 else 0.0 for i in range(64)]


def main(args):
    global binary
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix='heather-release-') as temp:
        temp = Path(temp)
        engine = Engine(temp / 'primary')
        engines = [engine]
        try:
            engine.start()
            route = '/db/audit/collections/records'
            engine.create('audit')
            engine.create('other')

            def auth_scope():
                expect(engine.request('GET', '/db', auth=False)[0] == 401, 'Unauthenticated database list accepted')
                expect(engine.request('GET', '/db', auth='Basic ' + base64.b64encode(b'admin:wrong').decode())[0] == 401, 'Wrong password accepted')
                password = secrets.token_urlsafe(20)
                created = engine.cli('user', 'create', 'scoped', '--password', password, '--scope', 'audit')
                expect(created.returncode == 0, 'Scoped user creation failed')
                auth = 'Basic ' + base64.b64encode(f'scoped:{password}'.encode()).decode()
                expect(engine.request('GET', '/db/audit/collections', auth=auth)[0] == 200, 'Scoped access failed')
                for path in ['/db', '/db/other/collections', '/collections', '/db/audit/audit']:
                    expect(engine.request('GET', path, auth=auth)[0] == 403, f'Scope bypass: {path}')
                status, token = engine.request('POST', '/auth/token', auth=auth)
                expect(status == 200, 'Token mint failed')
                bearer = 'Bearer ' + token['token']
                expect(engine.request('GET', '/db/other/collections', auth=bearer)[0] == 403, 'Bearer scope bypass')
                rotated = secrets.token_urlsafe(20)
                expect(engine.cli('user', 'passwd', 'scoped', '--password', rotated).returncode == 0, 'Password rotation failed')
                expect(engine.request('GET', '/db/audit/collections', auth=bearer)[0] == 401, 'Old token survived password rotation')
                auth = 'Basic ' + base64.b64encode(f'scoped:{rotated}'.encode()).decode()
                status, token = engine.request('POST', '/auth/token', auth=auth)
                expect(status == 200, 'Fresh token after rotation failed')
                bearer = 'Bearer ' + token['token']
                engine.cli('user', 'delete', 'scoped')
                expect(engine.request('GET', '/db/audit/collections', auth=auth)[0] == 401, 'Deleted user still authenticates with Basic')
                status = engine.request('GET', '/db/audit/collections', auth=bearer)[0]
                observations['deleted_user_bearer_http_status'] = status
                expect(status == 401, f'Deleted user bearer token still returns HTTP {status}')
            check('SEC-01 authentication, scope and deleted-user token revocation', auth_scope)

            def basic_write():
                status, payload = engine.request('POST', route + '/write', {'vectors': [unit(0), unit(1)], 'metadata': [{'label': 'keep'}, {'label': 'delete'}]})
                expect(status == 200 and len(payload['ids']) == 2, 'Metadata batch write failed')
                observations['ids'] = payload['ids']
                expect(engine.request('GET', route + '/documents/0')[1]['metadata']['label'] == 'keep', 'Document read mismatch')
                return {'records': 2}
            check('API-01 metadata write and exact document lookup', basic_write)

            def invalid_batch():
                before = engine.request('GET', route + '/stats')[1]
                for body in [{'vectors': [unit(2), [1, 2]], 'metadata': [{}, {}]}, {'vectors': [unit(2)], 'metadata': []}]:
                    expect(engine.request('POST', route + '/write', body)[0] == 400, 'Invalid batch accepted')
                after = engine.request('GET', route + '/stats')[1]
                expect(before == after, f'Invalid batch changed collection statistics: {before} vs {after}')
            check('API-02 invalid-item batch rollback', invalid_batch)

            def concurrent_reads_writes():
                def task(i):
                    if i % 2:
                        return engine.request('POST', route + '/write', {'vectors': [unit(i)], 'metadata': [{'sequence': i}]})[0]
                    return engine.request('POST', route + '/read', {'query': unit(0)})[0]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    statuses = list(pool.map(task, range(40)))
                expect(all(x == 200 for x in statuses), f'Concurrent operation failures: {statuses}')
                return {'operations': 40, 'clients': 4}
            check('API-03 bounded concurrent read/write workload', concurrent_reads_writes)

            def live_snapshot():
                out = temp / 'snapshot'
                snapshot = engine.cli('snapshot', 'create', '--db', 'audit', '--output', str(out))
                expect(snapshot.returncode == 0, 'Live snapshot failed: ' + snapshot.stderr[-200:])
                restored = Engine(temp / 'snapshot-restored')
                engines.append(restored)
                restore = engine.cli('restore', 'restore', '--input', str(out), directory=restored.directory)
                expect(restore.returncode == 0, 'Snapshot restore failed: ' + restore.stderr[-200:])
                restored.start()
                expect(restored.request('GET', route + '/documents/0')[1]['metadata']['label'] == 'keep', 'Restored snapshot missing record')
                restored.stop()
                expect(restored.cli('verify', '--json').returncode == 0, 'Restored snapshot failed integrity check')
            check('OPS-02 live snapshot, restore and verification', live_snapshot)

            def live_lock():
                second = engine.cli('verify', '--json')
                expect(second.returncode != 0, 'Offline verify accepted active engine directory')
                backup = engine.cli('backup', 'create', '--output', str(temp / 'live.tar.gz'))
                expect(backup.returncode != 0, 'Cold backup accepted active engine directory')
                process = subprocess.run([str(binary), '--data-dir', str(engine.directory), '--host', '127.0.0.1', '--port', '0'], capture_output=True, text=True, timeout=10)
                expect(process.returncode != 0, 'Second engine accepted data directory')
            check('OPS-01 exclusive ownership and offline-tool refusal', live_lock)

            def deletion():
                expect(engine.request('DELETE', route + '/documents/1')[0] == 200, 'Delete failed')
                expect(engine.request('GET', route + '/documents/1')[0] == 404, 'Deleted record retrievable')
            check('MEM-02 document deletion', deletion)

            def durability_restart():
                expect(engine.request('GET', '/db/audit/durability')[1]['sync'] is True, 'Default durability not synchronous')
                expect(engine.request('PUT', '/db/audit/durability', {'sync': False})[1]['sync'] is False, 'Relaxed mode toggle failed')
                expect(engine.request('PUT', '/db/audit/durability', {'sync': True})[1]['sync'] is True, 'Sync/flush toggle failed')
                engine.request('PUT', '/db/audit/durability', {'sync': False})
                engine.stop()
                engine.start()
                expect(engine.request('GET', '/db/audit/durability')[1]['sync'] is True, 'Restart did not restore sync')
            check('DUR-02 explicit durability modes, flush and restart default', durability_restart)

            def kill_restart():
                status, payload = engine.request('POST', route + '/write', {'vectors': [unit(42)], 'metadata': [{'durable': True}]})
                expect(status == 200, 'Durability sentinel write failed')
                identity = payload['ids'][0]
                engine.stop(kill=True)
                engine.start()
                expect(engine.request('GET', route + f'/documents/{identity}')[1]['metadata']['durable'] is True, 'Acknowledged write lost after SIGKILL')
                expect(engine.request('GET', route + '/documents/1')[0] == 404, 'Deleted record returned after restart')
            check('DUR-01 acknowledged synchronous write survives process kill', kill_restart)

            def cold_restore():
                engine.stop()
                try:
                    archive = temp / 'cold.tar.gz'
                    backup = engine.cli('backup', 'create', '--output', str(archive))
                    expect(backup.returncode == 0, 'Cold backup failed: ' + backup.stderr[-1000:])
                    restored = Engine(temp / 'cold-restored')
                    restored.password = engine.password
                    restored.auth = engine.auth
                    engines.append(restored)
                    restore = engine.cli('restore', 'restore', '--input', str(archive), directory=restored.directory)
                    expect(restore.returncode == 0, 'Cold restore failed: ' + restore.stderr[-1000:])
                    restored.start()
                    expect(restored.request('GET', route + '/documents/0')[0] == 200, 'Restored user/data could not be read')
                    restored.stop()
                    verify = restored.cli('verify', '--json')
                    expect(verify.returncode == 0, 'Cold restore integrity check failed: ' + verify.stderr[-1000:])
                finally:
                    # Later independent checks still need the primary server,
                    # even if backup, restore or verification fails.
                    engine.start()
            check('OPS-02 cold full backup restores data and users', cold_restore)

            def precision():
                for value in ['f64', 'f32']:
                    name = 'precision_' + value
                    engine.create(name, precision=value)
                    r = f'/db/{name}/collections/check'
                    expect(engine.request('POST', r + '/write', {'vectors': [unit(4)], 'metadata': [{'precision': value}]})[0] == 200, 'Precision write failed')
                    answer = engine.request('POST', r + '/documents/query', {'query': unit(4), 'n': 1})
                    expect(answer[0] == 200 and answer[1]['results'][0]['id'] == 0 and answer[1]['results'][0]['similarity'] > .99, 'Precision query failed')
            check('MEM-01 both storage precisions in plain mode', precision)

            def quality():
                engine.create('associations', competitive=True)
                rng = random.Random(20261105)
                patterns = []
                for _ in range(32):
                    v = [rng.gauss(0, 1) for _ in range(64)]
                    norm = math.sqrt(sum(x*x for x in v))
                    patterns.append([x/norm for x in v])
                r = '/db/associations/collections/memory'
                expect(engine.request('POST', r + '/write', {'vectors': patterns})[0] == 200, 'Associative corpus write failed')
                def cosine(a, b):
                    denominator = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
                    return sum(x*y for x, y in zip(a, b))/denominator if denominator else 0
                fidelities = []
                for v in patterns:
                    cue = [x if i % 4 else 0 for i, x in enumerate(v)]
                    status, answer = engine.request('POST', r + '/read', {'query': cue})
                    expect(status == 200, 'Partial-cue read failed')
                    fidelities.append(cosine(v, answer['result']))
                observations['associative_quality'] = {'seed': 20261105, 'patterns': 32, 'dimension': 64, 'erased_fraction': .25, 'mode': 'competitive, default beta=5, iterative', 'mean_cosine': statistics.mean(fidelities), 'minimum_cosine': min(fidelities), 'fraction_cosine_at_least_0_9': sum(x>=.9 for x in fidelities)/32, 'quality_threshold_is_not_a_release_guarantee': True}
                status, _ = engine.request('POST', '/db', {'name': 'sharp', 'dimension': 64, 'map_size_mb': 16, 'eam': {'competitive': True, 'beta': 50}})
                expect(status in (200,201), 'Configured associative database failed')
                sharp = '/db/sharp/collections/memory'
                expect(engine.request('POST', sharp+'/write', {'vectors': patterns})[0] == 200, 'Configured corpus write failed')
                controlled = []
                for v in patterns:
                    cue = [x if i % 4 else 0 for i,x in enumerate(v)]
                    status, answer = engine.request('POST', sharp+'/read', {'query':cue})
                    expect(status == 200, 'Configured recall failed')
                    controlled.append(cosine(v,answer['result']))
                observations['configured_associative_quality'] = {'seed':20261105, 'patterns':32, 'dimension':64, 'erased_fraction':.25, 'mode':'competitive, explicitly configured beta=50, iterative', 'mean_cosine':statistics.mean(controlled), 'minimum_cosine':min(controlled), 'fraction_cosine_at_least_0_9':sum(x>=.9 for x in controlled)/32, 'limits':'Synthetic fixture; beta=50 is a declared comparison, not a universal recommendation'}
                engine.create('recall_archive', competitive=False)
                archive = '/db/recall_archive/collections/memory'
                expect(engine.request('POST', archive+'/write', {'vectors': patterns})[0] == 200, 'Plain recall corpus failed')
                archived = []
                for v in patterns:
                    cue = [x if i % 4 else 0 for i,x in enumerate(v)]
                    status, answer = engine.request('POST', archive+'/attention', {'query':cue, 'scale':50})
                    expect(status == 200, 'Plain attention recall failed')
                    archived.append(cosine(v,answer['result']))
                observations['plain_attention_recall'] = {'seed':20261105, 'patterns':32, 'dimension':64, 'erased_fraction':.25, 'mode':'competitive=false, attention scale=50', 'mean_cosine':statistics.mean(archived), 'minimum_cosine':min(archived), 'fraction_cosine_at_least_0_9':sum(x>=.9 for x in archived)/32, 'fixture_tolerance': 'minimum cosine >= 0.9; this is not a universal capacity guarantee'}
                expect(min(archived) >= .9, 'Plain attention partial-cue recall fell below fixture tolerance')
                unknown = [rng.gauss(0,1) for _ in range(64)]
                _, answer = engine.request('POST', sharp+'/read', {'query':unknown})
                observations['unknown_query'] = {'finite_result':all(math.isfinite(x) for x in answer['result']), 'explicit_abstention':False, 'note':'The read API reconstructs a vector for novel input; no automatic unknown-query rejection is promised.'}
            check('MEM-03 deterministic partial-cue quality measurement', quality)

            def map_exhaustion():
                engine.create('bounded', map_size=1)
                r = '/db/bounded/collections/check'
                expect(engine.request('POST', r + '/write', {'vectors': [unit(0)], 'metadata': [{'sentinel': True}]})[0] == 200, 'Map sentinel write failed')
                before = engine.request('GET', r + '/stats')[1]
                status, error = engine.request('POST', r + '/write', {'vectors': [unit(i+1) for i in range(8)], 'metadata': [{'blob': 'x'*180000} for _ in range(8)]})
                expect(status == 500, f'Storage failure must return HTTP 500, got {status}')
                after = engine.request('GET', r + '/stats')[1]
                expect(engine.request('GET', r + '/documents/0')[0] == 200, 'Existing sentinel became unreadable')
                engine.stop()
                engine.start()
                reopened = engine.request('GET', r + '/stats')[1]
                observations['map_exhaustion'] = {'http_status': status, 'error': error, 'before': before, 'after_failed_write': after, 'after_restart': reopened}
                expect(before == after == reopened, 'Failed storage transaction changed live memory; see map_exhaustion evidence')
            check('API-02/OPS-03 map exhaustion leaves memory and disk unchanged', map_exhaustion)

            def future_format():
                engine.create('future')
                engine.stop()
                config = engine.directory / 'db/future/db.toml'
                source = config.read_text()
                import re
                config.write_text(re.sub(r'format_version = \d+', 'format_version = 999', source))
                before = config.read_bytes()
                engine.start()
                expect(engine.request('GET', '/db/future/collections')[0] == 409, 'Future format data access not refused')
                expect(config.read_bytes() == before, 'Refused format configuration rewritten')
                expect(engine.request('GET', '/db/audit')[0] == 200, 'Unrelated database unavailable')
                observations['ready_with_refused_database'] = engine.request('GET', '/ready')[0]
            check('FMT-01 unsupported database refused without rewriting', future_format)

            def binary_version():
                run = subprocess.run([str(binary), '--version'], capture_output=True, text=True, timeout=5)
                expect(run.returncode == 0, 'Binary has no successful --version command')
                return run.stdout.strip()
            check('REL-01 binary version discoverability', binary_version)
        finally:
            for instance in engines:
                instance.stop()
                if instance.stderr is not None:
                    instance.stderr.close()

    report = {'recordedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'engineCommit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), 'checks': checks, 'observations': observations, 'limits': [f'{platform.system()} {platform.machine()} run; not power-loss certification', '40-operation concurrency check is not a soak or sustained load test', 'No pre-1.0 migration tested or required', 'No remote services or existing data directories used']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': sum(x['status']=='pass' for x in checks), 'failed': sum(x['status']=='fail' for x in checks)}))
    return 1 if any(x['status']=='fail' for x in checks) else 0


if __name__ == '__main__':
    raise SystemExit(main(parser.parse_args()))
