#!/usr/bin/env python3
"""Train an EAMClassifier on MNIST and export for HeatherDB.

Downloads MNIST, trains the controller (Phase 1), then runs continual
learning (Phase 2) to populate the EAM. Exports the result as JSON +
controller checkpoint for use with heather-fornix.

Usage:
    python train.py

Output:
    mnist.json              — EAM state + config + prototypes
    mnist_controller.pt     — encoder weights
"""

import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from synapse.model import EAMClassifier
from synapse.tasks import Task
from synapse.training import train_controller, continual_learning_eval


def export_eam(model: EAMClassifier, path: str) -> str:
    """Export EAM state to HeatherDB-compatible JSON + controller checkpoint."""
    mem = model.memory
    out = Path(path)
    num_locs = mem.addresses.shape[0]

    # Map local EAMLayer attrs to HeatherDB's EAMConfig.
    # The local layer is simplified, so use defaults for missing fields.
    config = {
        "d": mem.dim,
        "l_0": num_locs,
        "l_max": num_locs * 2,
        "k": mem.k,
        "eta_0": mem.eta,
        "lambda": 0.9999,
        "eta_min": 0.001,
        "tau_split": 0.3,
        "tau_merge": 0.95,
        "gamma": 1.0,
        "tau_damp": 10.0,
        "tau_overload": 100.0,
        "beta": mem.beta,
        "t_max": mem.t_max,
        "epsilon": 1e-6,
    }

    addresses = mem.addresses.detach().cpu().double()
    counters = mem.counters.detach().cpu().double()
    write_counts = mem.write_counts.detach().cpu().double()

    locations = [
        {
            "id": i,
            "address": addresses[i].tolist(),
            "counter": counters[i].tolist(),
            "write_count": float(write_counts[i]),
        }
        for i in range(num_locs)
    ]

    prototypes = {
        str(label): proto.detach().cpu().double().tolist()
        for label, proto in model._prototypes.items()
    }

    ctrl_path = out.with_name(out.stem + "_controller.pt")
    torch.save(model.controller.state_dict(), ctrl_path)

    export = {
        "config": config,
        "locations": locations,
        "prototypes": prototypes,
        "controller_state": ctrl_path.name,
    }
    out.write_text(json.dumps(export))
    return str(out)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download MNIST and return flattened float tensors."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # 28x28 -> 784
    ])

    train_ds = datasets.MNIST("./data_mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data_mnist", train=False, download=True, transform=transform)

    train_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    return train_images, train_labels, test_images, test_labels


def make_tasks(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    classes_per_task: int = 2,
    exemplars_per_class: int = 50,
    train_per_class: int = 500,
    test_per_class: int = 200,
) -> list[Task]:
    """Split MNIST digits into sequential tasks (strata Task format)."""
    all_classes = sorted(train_labels.unique().tolist())
    tasks = []

    for task_id, start in enumerate(range(0, len(all_classes), classes_per_task)):
        task_classes = all_classes[start : start + classes_per_task]
        if not task_classes:
            break

        exemplar_list, exemplar_label_list = [], []
        train_list, train_label_list = [], []
        test_list, test_label_list = [], []

        for cls in task_classes:
            # Training data for this class
            tr_mask = train_labels == cls
            tr_imgs = train_images[tr_mask]
            perm = torch.randperm(tr_imgs.shape[0])

            exemplar_list.append(tr_imgs[perm[:exemplars_per_class]])
            exemplar_label_list.append(torch.full((exemplars_per_class,), cls))

            train_list.append(tr_imgs[perm[exemplars_per_class : exemplars_per_class + train_per_class]])
            train_label_list.append(torch.full((train_per_class,), cls))

            # Test data for this class
            te_mask = test_labels == cls
            te_imgs = test_images[te_mask]
            te_perm = torch.randperm(te_imgs.shape[0])
            test_list.append(te_imgs[te_perm[:test_per_class]])
            test_label_list.append(torch.full((test_per_class,), cls))

        tasks.append(Task(
            task_id=task_id,
            class_labels=task_classes,
            exemplars=torch.cat(exemplar_list),
            exemplar_labels=torch.cat(exemplar_label_list),
            train_inputs=torch.cat(train_list),
            train_labels=torch.cat(train_label_list),
            test_inputs=torch.cat(test_list),
            test_labels=torch.cat(test_label_list),
        ))

    return tasks


def main():
    print("=== Synapse: MNIST Training ===\n")

    print("Loading MNIST...", flush=True)
    train_images, train_labels, test_images, test_labels = load_mnist()
    print(f"  Train: {train_images.shape[0]} images")
    print(f"  Test:  {test_images.shape[0]} images\n")

    print("Creating tasks (2 digits per task, 5 tasks total)...\n")
    tasks = make_tasks(train_images, train_labels, test_images, test_labels)
    for t in tasks:
        print(f"  Task {t.task_id}: digits {t.class_labels}")

    # Build model
    model = EAMClassifier(
        input_dim=784,
        hidden_dim=128,
        memory_dim=64,
        num_locations=500,
        k=20,
        beta=5.0,
        t_max=3,
    )

    # Phase 1: Train controller
    print("\n--- Phase 1: Training controller ---\n")
    history = train_controller(model, tasks, epochs=50, lr=1e-3, batch_size=64)
    final = history[-1]
    print(f"\nFinal: loss={final['loss']:.4f}, accuracy={final['accuracy']:.1%}\n")

    # Phase 2: Continual learning (freeze controller, populate EAM)
    print("--- Phase 2: Continual learning ---\n")
    results = continual_learning_eval(model, tasks)
    final_avg = results[-1]["avg_accuracy"]
    print(f"\nFinal avg accuracy across all tasks: {final_avg:.1%}\n")

    # Export
    out_path = Path(__file__).parent / "mnist.json"
    print(f"Exporting to {out_path}...")
    export_eam(model, str(out_path))
    print(f"  Created: mnist.json + mnist_controller.pt")
    print(f"  Locations: {model.memory.num_locations}")
    print(f"  Prototypes: {sorted(model._prototypes.keys())}")
    print(f"\nNext steps:")
    print(f"  1. cargo run -p heather_fornix -- import -f {out_path} -c mnist -d ./data")
    print(f"  2. cargo run -p heather_server -- --dimension 64 --data-dir ./data")
    print(f"  3. python run.py")


if __name__ == "__main__":
    main()
