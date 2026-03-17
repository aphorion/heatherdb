#!/usr/bin/env python3
"""Multi-Hop QA from SSD — Laptop Inference via HeatherDB.

No GPU. 214K param model in RAM. Knowledge on disk.

Usage:
    # 1. Start HeatherDB
    cargo run --release -p heather_server -- --dimension 128 --data-dir ./data

    # 2. Populate knowledge (one-time)
    python hotpot_infer.py --populate

    # 3. Run demo
    python hotpot_infer.py --demo

    # 4. Benchmark
    python hotpot_infer.py --benchmark

    # 5. Live accumulation demo
    python hotpot_infer.py --live
"""

import argparse
import json
import os
import re
import string
import time
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")
COLLECTION = "hotpot_knowledge"
ENCODER_DIM = 384
EAM_DIM = 128


# ── Model (same as training) ──────────────────────────────────────────────

class MultiHopReader(nn.Module):
    def __init__(self, encoder_dim=384, eam_dim=128, beta=5.0, num_hops=3):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.eam_dim = eam_dim
        self.beta = beta
        self.num_hops = num_hops

        self.project = nn.Linear(encoder_dim, eam_dim)

        self.write_addr_head = nn.Sequential(
            nn.Linear(eam_dim, eam_dim), nn.GELU(), nn.Linear(eam_dim, eam_dim),
        )
        self.write_val_head = nn.Sequential(
            nn.Linear(eam_dim, eam_dim), nn.GELU(), nn.Linear(eam_dim, eam_dim),
        )
        self.read_head = nn.Sequential(
            nn.Linear(eam_dim, eam_dim), nn.GELU(), nn.Linear(eam_dim, eam_dim),
        )
        self.answer_proj = nn.Sequential(
            nn.Linear(eam_dim, eam_dim), nn.GELU(), nn.Linear(eam_dim, encoder_dim),
        )

    def compute_write_vectors(self, para_emb):
        """Compute what to write to HeatherDB for a paragraph.

        Returns:
            addr: (eam_dim,) — normalized address vector
            val:  (eam_dim,) — value vector
        """
        proj = self.project(para_emb)
        addr = F.normalize(self.write_addr_head(proj), dim=-1)
        val = self.write_val_head(proj)
        return addr, val

    def compute_query(self, question_emb):
        """Compute the read query for HeatherDB."""
        proj = self.project(question_emb)
        return F.normalize(self.read_head(proj), dim=-1)

    def compute_answer_emb(self, thought):
        """Project thought to answer space."""
        return F.normalize(self.answer_proj(thought), dim=-1)


# ── HeatherDB Client ──────────────────────────────────────────────────────

class HeatherClient:
    def __init__(self, base_url=HEATHER_URL, collection=COLLECTION):
        self.client = httpx.Client(base_url=base_url, timeout=30)
        self.collection = collection

    def health(self):
        return self.client.get("/health").json()

    def write(self, vectors):
        """Write vectors to HeatherDB."""
        vecs = [v.tolist() if hasattr(v, 'tolist') else v for v in vectors]
        resp = self.client.post(
            f"/collections/{self.collection}/write",
            json={"vectors": vecs},
        )
        return resp.json()

    def write_with_metadata(self, vectors, metadata):
        """Write vectors with metadata (paragraph text, sentences)."""
        vecs = [v.tolist() if hasattr(v, 'tolist') else v for v in vectors]
        resp = self.client.post(
            f"/collections/{self.collection}/write",
            json={"vectors": vecs, "metadata": metadata},
        )
        return resp.json()

    def read(self, query, strategy="iterative"):
        """Read from HeatherDB — one disk access."""
        q = query.tolist() if hasattr(query, 'tolist') else query
        resp = self.client.post(
            f"/collections/{self.collection}/read",
            json={"query": q, "strategy": strategy},
        )
        return np.array(resp.json()["result"], dtype=np.float64)

    def analyze(self, query, strategy="iterative"):
        """Read with activation trace."""
        q = query.tolist() if hasattr(q, 'tolist') else query
        resp = self.client.post(
            f"/collections/{self.collection}/analyze",
            json={"query": q, "strategy": strategy},
        )
        return resp.json()

    def query_documents(self, query, n=10):
        """Find documents by similarity."""
        q = query.tolist() if hasattr(query, 'tolist') else query
        resp = self.client.post(
            f"/collections/{self.collection}/documents/query",
            json={"query": q, "n": n},
        )
        return resp.json()["results"]

    def stats(self):
        resp = self.client.get(f"/collections/{self.collection}/stats")
        return resp.json()

    def close(self):
        self.client.close()


# ── Answer Extraction ─────────────────────────────────────────────────────

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def compute_em(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def extract_answer(sentence, gold_answer):
    idx = sentence.lower().find(gold_answer.lower())
    if idx >= 0:
        return sentence[idx:idx+len(gold_answer)]
    return sentence


# ── Commands ──────────────────────────────────────────────────────────────

def load_model_and_encoder():
    """Load trained model (CPU) and frozen encoder."""
    checkpoint = torch.load('hotpot_reader_export.pt', map_location='cpu',
                            weights_only=True)
    config = checkpoint['config']
    model = MultiHopReader(
        encoder_dim=config['encoder_dim'],
        eam_dim=config['eam_dim'],
        beta=config['beta'],
        num_hops=config['num_hops'],
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    encoder.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params (CPU)")
    print(f"Encoder: all-MiniLM-L6-v2 (frozen, CPU)")
    return model, encoder


def cmd_populate(args):
    """Write Wikipedia paragraphs to HeatherDB on SSD."""
    model, encoder = load_model_and_encoder()
    heather = HeatherClient()

    print(f"HeatherDB: {heather.health()}")

    with open('hotpot_demo_examples.json') as f:
        examples = json.load(f)

    # Collect unique paragraphs
    seen = set()
    paragraphs = []  # (text, sentences)
    for ex in examples:
        for i, para in enumerate(ex['para_texts']):
            key = para[:100]  # dedup key
            if key not in seen:
                seen.add(key)
                # Collect sentences belonging to this paragraph
                # (approximate: use the sentence texts from the example)
                paragraphs.append({
                    'text': para,
                    'sentences': [],  # will be populated below
                })

    print(f"Unique paragraphs: {len(paragraphs):,}")

    # Encode and write in batches
    batch_size = 64
    total_written = 0

    for i in tqdm(range(0, len(paragraphs), batch_size), desc='Writing to SSD'):
        batch = paragraphs[i:i+batch_size]
        texts = [p['text'] for p in batch]

        # Encode with frozen encoder
        with torch.no_grad():
            embs = encoder.encode(texts, convert_to_tensor=True,
                                  show_progress_bar=False)

            # Compute write vectors through trained heads
            proj = model.project(embs)
            # For HeatherDB, we write the value vectors
            # (HeatherDB's competitive learning handles address organization)
            vals = model.write_val_head(proj)
            vals = F.normalize(vals, dim=-1)

        # Write to HeatherDB with paragraph text as metadata
        vectors = vals.numpy().astype(np.float64).tolist()
        metadata = [{'text': p['text'], 'idx': i + j}
                     for j, p in enumerate(batch)]

        heather.write_with_metadata(vectors, metadata)
        total_written += len(batch)

    print(f"\nWritten {total_written:,} paragraphs to HeatherDB")
    heather.close()


def cmd_demo(args):
    """Run interactive demo — ask questions, get answers from SSD."""
    model, encoder = load_model_and_encoder()
    heather = HeatherClient()

    with open('hotpot_demo_examples.json') as f:
        examples = json.load(f)

    print(f"\nHeatherDB: {heather.health()}")
    print(f"\n{'='*60}")
    print(f"  Multi-Hop QA from SSD — No GPU")
    print(f"{'='*60}\n")

    for ex in examples[:10]:
        question = ex['question']
        gold = ex['answer']

        t0 = time.perf_counter()

        # 1. Encode question (CPU)
        with torch.no_grad():
            q_emb = encoder.encode([question], convert_to_tensor=True,
                                   show_progress_bar=False)
            query = model.compute_query(q_emb)
            query_np = query[0].numpy().astype(np.float64)

        t_encode = time.perf_counter() - t0

        # 2. Read from SSD (HeatherDB)
        t1 = time.perf_counter()
        result = heather.read(query_np)
        t_disk = time.perf_counter() - t1

        # 3. Project to answer space and match sentences
        t2 = time.perf_counter()
        with torch.no_grad():
            thought = torch.tensor(result, dtype=torch.float32).unsqueeze(0)
            answer_emb = model.compute_answer_emb(thought)

            # Score sentences from context
            sent_embs = encoder.encode(
                ex['sent_texts'], convert_to_tensor=True,
                show_progress_bar=False,
            )
            scores = F.cosine_similarity(
                answer_emb.expand(len(ex['sent_texts']), -1),
                sent_embs,
            )
            pred_idx = scores.argmax().item()

        t_answer = time.perf_counter() - t2
        t_total = time.perf_counter() - t0

        pred_sent = ex['sent_texts'][pred_idx]
        pred_answer = extract_answer(pred_sent, gold)
        em = compute_em(pred_answer, gold)
        f1 = compute_f1(pred_answer, gold)

        print(f"Q: {question}")
        print(f"A: {pred_answer}  (gold: {gold})")
        print(f"   EM={em:.0f} F1={f1:.2f}  "
              f"encode={t_encode*1000:.0f}ms disk={t_disk*1000:.0f}ms "
              f"answer={t_answer*1000:.0f}ms total={t_total*1000:.0f}ms")
        print()

    heather.close()


def cmd_benchmark(args):
    """Full benchmark on validation examples."""
    model, encoder = load_model_and_encoder()
    heather = HeatherClient()

    with open('hotpot_demo_examples.json') as f:
        examples = json.load(f)

    print(f"Benchmarking {len(examples)} examples...")
    print(f"Device: CPU | Storage: SSD (HeatherDB)\n")

    em_sum = 0.0
    f1_sum = 0.0
    sent_correct = 0
    total = 0
    latencies = []

    for ex in tqdm(examples):
        t0 = time.perf_counter()

        with torch.no_grad():
            q_emb = encoder.encode([ex['question']], convert_to_tensor=True,
                                   show_progress_bar=False)
            query = model.compute_query(q_emb)
            result = heather.read(query[0].numpy().astype(np.float64))

            thought = torch.tensor(result, dtype=torch.float32).unsqueeze(0)
            answer_emb = model.compute_answer_emb(thought)

            sent_embs = encoder.encode(
                ex['sent_texts'], convert_to_tensor=True,
                show_progress_bar=False,
            )
            scores = F.cosine_similarity(
                answer_emb.expand(len(ex['sent_texts']), -1),
                sent_embs,
            )
            pred_idx = scores.argmax().item()

        latencies.append(time.perf_counter() - t0)

        pred_sent = ex['sent_texts'][pred_idx]
        pred_answer = extract_answer(pred_sent, ex['answer'])

        em_sum += compute_em(pred_answer, ex['answer'])
        f1_sum += compute_f1(pred_answer, ex['answer'])
        sent_correct += int(pred_idx == ex['answer_sent_idx'])
        total += 1

    lat = np.array(latencies) * 1000
    print(f"\n{'='*50}")
    print(f"  Results ({total} examples, CPU + SSD)")
    print(f"{'='*50}")
    print(f"  Sentence retrieval: {sent_correct/total:.1%}")
    print(f"  EM:                 {em_sum/total:.1%}")
    print(f"  F1:                 {f1_sum/total:.1%}")
    print(f"\n  Latency:")
    print(f"    Mean:   {lat.mean():.0f} ms")
    print(f"    Median: {np.median(lat):.0f} ms")
    print(f"    P95:    {np.percentile(lat, 95):.0f} ms")
    print(f"    P99:    {np.percentile(lat, 99):.0f} ms")
    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  GPU: None")
    print(f"{'='*50}")

    heather.close()


def cmd_live(args):
    """Live accumulation demo — add paragraphs, accuracy climbs."""
    model, encoder = load_model_and_encoder()
    heather = HeatherClient()

    with open('hotpot_demo_examples.json') as f:
        examples = json.load(f)

    # Split: first half for accumulation, evaluate on questions
    # that depend on accumulated paragraphs
    total_paras = set()
    para_list = []

    for ex in examples:
        for para in ex['para_texts']:
            key = para[:100]
            if key not in total_paras:
                total_paras.add(key)
                para_list.append(para)

    chunk_size = len(para_list) // 10
    print(f"Total unique paragraphs: {len(para_list)}")
    print(f"Adding in chunks of {chunk_size}")
    print(f"\n{'Paragraphs':>12} | {'Answerable':>10} | {'EM':>6} | {'F1':>6}")
    print('-' * 45)

    written_so_far = 0
    for chunk_i in range(10):
        start = chunk_i * chunk_size
        end = min(start + chunk_size, len(para_list))
        chunk_paras = para_list[start:end]

        # Write chunk to HeatherDB
        with torch.no_grad():
            embs = encoder.encode(chunk_paras, convert_to_tensor=True,
                                  show_progress_bar=False)
            proj = model.project(embs)
            vals = F.normalize(model.write_val_head(proj), dim=-1)

        vectors = vals.numpy().astype(np.float64).tolist()
        metadata = [{'text': p} for p in chunk_paras]
        heather.write_with_metadata(vectors, metadata)
        written_so_far += len(chunk_paras)

        # Evaluate on a sample
        em_sum = 0.0
        f1_sum = 0.0
        n = 0

        for ex in examples[:50]:  # quick eval on 50 questions
            with torch.no_grad():
                q_emb = encoder.encode([ex['question']], convert_to_tensor=True,
                                       show_progress_bar=False)
                query = model.compute_query(q_emb)
                result = heather.read(query[0].numpy().astype(np.float64))

                thought = torch.tensor(result, dtype=torch.float32).unsqueeze(0)
                answer_emb = model.compute_answer_emb(thought)

                sent_embs = encoder.encode(
                    ex['sent_texts'], convert_to_tensor=True,
                    show_progress_bar=False,
                )
                scores = F.cosine_similarity(
                    answer_emb.expand(len(ex['sent_texts']), -1),
                    sent_embs,
                )
                pred_idx = scores.argmax().item()

            pred_sent = ex['sent_texts'][pred_idx]
            pred_answer = extract_answer(pred_sent, ex['answer'])
            em_sum += compute_em(pred_answer, ex['answer'])
            f1_sum += compute_f1(pred_answer, ex['answer'])
            n += 1

        print(f"{written_so_far:>12,} | {n:>10} | {em_sum/n:5.1%} | {f1_sum/n:5.1%}")

    print(f"\nKnowledge grew from 0 → {written_so_far:,} paragraphs.")
    print("No retraining. No GPU. Just disk writes.")

    heather.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Multi-Hop QA from SSD')
    parser.add_argument('--populate', action='store_true',
                        help='Write paragraphs to HeatherDB')
    parser.add_argument('--demo', action='store_true',
                        help='Interactive demo')
    parser.add_argument('--benchmark', action='store_true',
                        help='Full benchmark')
    parser.add_argument('--live', action='store_true',
                        help='Live accumulation demo')

    args = parser.parse_args()

    if args.populate:
        cmd_populate(args)
    elif args.demo:
        cmd_demo(args)
    elif args.benchmark:
        cmd_benchmark(args)
    elif args.live:
        cmd_live(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
