#!/usr/bin/env python3
"""Extract a release archive, verify its version, and smoke-test persistent HTTP."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


def smoke(binary, version, root):
    actual = subprocess.check_output([str(binary), '--version'], text=True).strip()
    assert actual == f'heather {version}', actual
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    password = secrets.token_urlsafe(24)
    auth = 'Basic ' + base64.b64encode(f'admin:{password}'.encode()).decode()
    env = {k: v for k, v in os.environ.items() if not k.startswith('HEATHER_')}
    env.update(HEATHER_ADMIN_USER='admin', HEATHER_ADMIN_PASSWORD=password, RUST_LOG='error')
    command = [str(binary), '--data-dir', str(root / 'data'), '--host', '127.0.0.1', '--port', str(port), '--dimension', '8', '--map-size-mb', '32']

    def request(method, route, data=None, authenticated=True):
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers['Authorization'] = auth
        req = urllib.request.Request(f'http://127.0.0.1:{port}{route}', None if data is None else json.dumps(data).encode(), headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, None

    for restart in [False, True]:
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                for _ in range(200):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError('Startup failed: ' + log.read().decode(errors='replace')[-2000:])
                    try:
                        if request('GET', '/ready', authenticated=False)[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise TimeoutError('Packaged binary did not become ready')
                assert request('GET', '/db', authenticated=False)[0] == 401
                if not restart:
                    status, body = request('POST', '/collections/release_smoke/write', {'vectors': [[1, 0, 0, 0, 0, 0, 0, 0]], 'metadata': [{'release': version}]})
                    assert status == 200 and body['ids'] == [0], (status, body)
                    assert request('POST', '/collections/release_smoke/read', {'query': [1, 0, 0, 0, 0, 0, 0, 0]})[0] == 200
                status, body = request('GET', '/collections/release_smoke/documents/0')
                assert status == 200 and body['metadata']['release'] == version
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    assert bool(args.archive) != bool(args.binary), 'Choose --archive or --binary'
    with tempfile.TemporaryDirectory(prefix='heather-package-') as temp:
        root = Path(temp)
        if args.archive:
            archive = args.archive.resolve()
            target = root / 'unpacked'
            target.mkdir()
            if archive.suffix == '.zip':
                with zipfile.ZipFile(archive) as package:
                    for member in package.infolist():
                        dest = (target / member.filename).resolve()
                        assert dest.is_relative_to(target), member.filename
                    package.extractall(target)
            else:
                with tarfile.open(archive, 'r:gz') as package:
                    for member in package.getmembers():
                        dest = (target / member.name).resolve()
                        assert dest.is_relative_to(target) and (member.isfile() or member.isdir()), member.name
                    package.extractall(target, filter='data')
            binaries = [p for p in target.rglob('*') if p.is_file() and p.name in ['heather', 'heather.exe']]
            assert len(binaries) == 1, binaries
            binary = binaries[0]
            binary.chmod(0o755)
        else:
            binary = args.binary.resolve()
        smoke(binary, args.version.removeprefix('v'), root)
        result = {'version': args.version.removeprefix('v'), 'binarySha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'checks': ['version', 'startup', 'authentication', 'write', 'read', 'document lookup', 'restart persistence']}
        if args.archive:
            result.update(archive=archive.name, archiveSha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        print(json.dumps(result))


if __name__ == '__main__':
    main()
