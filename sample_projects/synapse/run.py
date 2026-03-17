#!/usr/bin/env python3
"""Classify MNIST digits through HeatherDB.

Loads the encoder (controller) and decoder (prototypes) in local memory.
The EAM lives server-side in HeatherDB. Each classification sends an
encoded query to HeatherDB's /read endpoint, then matches the
reconstruction against prototypes.

Prerequisites:
    1. python train.py                          (train + export)
    2. heather-fornix import -f mnist.json ...  (import EAM into HeatherDB)
    3. heather_server running with the data     (serve the EAM)
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from synapse.client import HeatherClient
from synapse.classifier import MNISTClassifier


def render_digit(pixels: torch.Tensor, width: int = 28) -> str:
    """Render a flattened MNIST image as ASCII art."""
    chars = " .:-=+*#%@"
    img = pixels.view(width, width).numpy()
    lines = []
    for row in img:
        line = ""
        for val in row:
            idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx] * 2  # double width for aspect ratio
        lines.append(line)
    return "\n".join(lines)


def main():
    print("=== Synapse: MNIST Classification via HeatherDB ===\n")

    export_path = Path(__file__).parent / "mnist.json"
    if not export_path.exists():
        print("mnist.json not found. Run train.py first.")
        sys.exit(1)

    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    collection = os.getenv("COLLECTION", "mnist")

    heather = HeatherClient(url=url, collection=collection)

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --dimension 64 --data-dir ./data")
        sys.exit(1)
    print("OK")

    stats = heather.stats()
    print(f"  Collection '{collection}': {stats['num_locations']} locations, "
          f"{stats['total_writes']:.0f} total writes\n")

    # Load classifier (encoder + decoder in memory)
    print("Loading encoder + prototypes...", end=" ", flush=True)
    classifier = MNISTClassifier(str(export_path), heather)
    print(f"OK ({len(classifier.labels)} classes: {classifier.labels})\n")

    # Load test set
    print("Loading MNIST test set...", end=" ", flush=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    test_ds = datasets.MNIST(
        "./data_mnist", train=False, download=True, transform=transform
    )
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    print(f"{len(test_ds)} images\n")

    # Demo: classify 10 random digits
    print("--- Random Samples ---\n")
    indices = random.sample(range(len(test_ds)), 10)

    correct = 0
    for idx in indices:
        image = test_images[idx]
        true_label = int(test_labels[idx])

        pred_label, confidence = classifier.classify(image)
        is_correct = pred_label == true_label
        correct += int(is_correct)

        mark = "OK" if is_correct else "WRONG"
        print(render_digit(image))
        print(f"  True: {true_label}  Predicted: {pred_label}  "
              f"Confidence: {confidence:.3f}  [{mark}]\n")

    print(f"Sample accuracy: {correct}/{len(indices)}\n")

    # Full test set evaluation
    print("--- Full Test Evaluation ---\n")
    n_test = min(1000, len(test_ds))
    print(f"Classifying {n_test} test images...", flush=True)

    perm = torch.randperm(len(test_ds))[:n_test]
    batch_images = test_images[perm]
    batch_labels = test_labels[perm]

    results = classifier.classify_batch(batch_images)

    total_correct = sum(
        1 for (pred, _), true in zip(results, batch_labels.tolist())
        if pred == true
    )
    accuracy = total_correct / n_test

    print(f"\nAccuracy: {total_correct}/{n_test} = {accuracy:.1%}")

    # Per-digit breakdown
    print("\nPer-digit accuracy:")
    for digit in range(10):
        mask = batch_labels == digit
        if mask.sum() == 0:
            continue
        digit_correct = sum(
            1 for i, (pred, _) in enumerate(results)
            if mask[i] and pred == digit
        )
        digit_total = int(mask.sum())
        print(f"  {digit}: {digit_correct}/{digit_total} = {digit_correct / digit_total:.1%}")

    heather.close()


if __name__ == "__main__":
    main()
