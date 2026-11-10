#!/usr/bin/env python3
"""Replay saved competitive attention readouts and verify retained artifacts."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.results.read_text())
for item in report['artifacts']:
    file = args.artifacts / item['file']
    assert file.stat().st_size == item['bytes'], file
    assert hashlib.sha256(file.read_bytes()).hexdigest() == item['sha256'], file
replayed = 0
for row in report['learning']:
    file = args.artifacts / f"{row['condition']}_s{row['seed']}_n{row['writes']}.npz"
    with np.load(file) as saved:
        assert hashlib.sha256(np.ascontiguousarray(saved['train']).tobytes()).hexdigest() == row['data_hash']
        assert hashlib.sha256(np.ascontiguousarray(saved['test']).tobytes()).hexdigest() == row['test_hash']
        training = {value.tobytes() for value in saved['train']}
        assert not any(value.tobytes() in training for value in saved['test'])
        targets = saved['prototypes'][saved['test_labels']]
        for tag in ['default', 'winner']:
            keys, values, counts = (saved[f'{tag}_{name}'] for name in ['keys', 'values', 'counts'])
            scores = []
            for query, target in zip(saved['test'], targets):
                dots = keys @ query
                indices = np.argsort(-dots, kind='stable')[:20]
                logits = 50 * dots[indices] + np.log(np.maximum(counts[indices], 1e-30))
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                answer = np.sum(values[indices] * weights[:, None], axis=0)
                score = np.dot(answer, target) / max(np.linalg.norm(answer) * np.linalg.norm(target), 1e-30)
                scores.append(score)
            np.testing.assert_allclose(scores, saved[f'competitive_{tag}_attention_50'], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(np.mean(scores), row['measurements'][f'competitive_{tag}']['attention_50_cosine'], atol=1e-12, rtol=1e-12)
            replayed += 1
print(json.dumps({'artifacts_verified': len(report['artifacts']), 'readouts_replayed': replayed, 'train_test_disjoint': True}))
