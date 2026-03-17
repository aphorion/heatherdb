"""Training loop, evaluation, and baseline comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import EAMClassifier
from .tasks import Task


def train_controller(
    model: EAMClassifier,
    tasks: list[Task],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> list[dict]:
    """Phase 1: Train the controller's encoding strategy.

    Uses DIRECT prototype matching (no EAM read) so the controller learns
    a clean metric embedding. The EAM is populated after training completes.

    This trains HOW to encode — the WHAT (EAM contents) comes later.

    Returns: list of {epoch, loss, accuracy} dicts.
    """
    # Register exemplars (builds prototypes, no EAM write yet)
    all_exemplar_x = torch.cat([t.exemplars for t in tasks])
    all_exemplar_y = torch.cat([t.exemplar_labels for t in tasks])

    # Bootstrap prototypes from initial controller encoding
    with torch.no_grad():
        for task in tasks:
            encoded = model.controller(task.exemplars)
            for label_val in task.exemplar_labels.unique():
                mask = task.exemplar_labels == label_val
                key = label_val.item()
                model._prototypes[key] = encoded[mask].mean(dim=0)
                model._prototype_counts[key] = mask.sum().item()
                model._exemplar_inputs[key] = task.exemplars[mask].clone()

    optimizer = torch.optim.Adam(model.controller.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    all_train_x = torch.cat([t.train_inputs for t in tasks])
    all_train_y = torch.cat([t.train_labels for t in tasks])

    history = []
    for epoch in range(epochs):
        # Refresh prototypes from current controller each epoch
        model.refresh_prototypes()

        label_to_idx = {lab: idx for idx, lab in enumerate(model.class_labels)}

        perm = torch.randperm(all_train_x.shape[0])
        x_shuf = all_train_x[perm]
        y_shuf = all_train_y[perm]

        total_loss = 0.0
        correct = 0
        total = 0

        model.controller.train()
        for i in range(0, len(x_shuf), batch_size):
            bx = x_shuf[i : i + batch_size]
            by = y_shuf[i : i + batch_size]
            by_idx = torch.tensor([label_to_idx[y.item()] for y in by])

            optimizer.zero_grad()
            logits = model(bx, use_memory=False)  # direct prototype matching
            loss = criterion(logits, by_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.shape[0]
            correct += (logits.argmax(dim=1) == by_idx).sum().item()
            total += bx.shape[0]

        avg_loss = total_loss / total
        accuracy = correct / total
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")

    return history


def evaluate(model: EAMClassifier, tasks: list[Task]) -> dict[int, float]:
    """Evaluate per-task accuracy on test sets (using EAM read).

    Handles tasks whose classes are not in the model's prototypes
    (e.g., after knowledge surgery) by reporting 0% accuracy.
    """
    model.controller.eval()
    known_labels = set(model.class_labels)
    label_to_idx = {lab: idx for idx, lab in enumerate(model.class_labels)}
    results = {}

    for task in tasks:
        # Skip tasks whose classes aren't in the prototype set
        if not all(c in known_labels for c in task.class_labels):
            results[task.task_id] = 0.0
            continue

        with torch.no_grad():
            logits = model(task.test_inputs, use_memory=True)
        by_idx = torch.tensor([label_to_idx[y.item()] for y in task.test_labels])
        acc = (logits.argmax(dim=1) == by_idx).float().mean().item()
        results[task.task_id] = acc

    return results


def continual_learning_eval(
    model: EAMClassifier,
    all_tasks: list[Task],
) -> list[dict]:
    """Phase 2: Freeze controller, clear EAM, stream tasks.

    After each task's exemplars are written, evaluate on ALL tasks seen so far.
    Uses EAM read for classification — this is where WHAT matters.
    """
    for param in model.controller.parameters():
        param.requires_grad_(False)
    model.controller.eval()

    model.clear_memory()

    results = []
    tasks_seen = []

    for task in all_tasks:
        tasks_seen.append(task)
        model.write_exemplars(task.exemplars, task.exemplar_labels)

        accuracies = evaluate(model, tasks_seen)
        avg_acc = sum(accuracies.values()) / len(accuracies)
        results.append({
            "after_task": task.task_id,
            "accuracies": accuracies,
            "avg_accuracy": avg_acc,
        })

        acc_str = ", ".join(
            f"T{tid}={acc:.0%}" for tid, acc in sorted(accuracies.items())
        )
        print(f"  After Task {task.task_id}: {acc_str}  (avg={avg_acc:.0%})")

    return results


def baseline_mlp_eval(
    all_tasks: list[Task],
    input_dim: int = 128,
    hidden_dim: int = 128,
    epochs_per_task: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> list[dict]:
    """Phase 3: Standard MLP trained sequentially — shows catastrophic forgetting."""
    max_label = 0
    for t in all_tasks:
        max_label = max(max_label, max(t.class_labels))
    num_classes = max_label + 1

    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    results = []
    tasks_seen = []

    for task in all_tasks:
        tasks_seen.append(task)

        mlp.train()
        for _ in range(epochs_per_task):
            perm = torch.randperm(task.train_inputs.shape[0])
            x = task.train_inputs[perm]
            y = task.train_labels[perm]
            for i in range(0, len(x), batch_size):
                optimizer.zero_grad()
                logits = mlp(x[i : i + batch_size])
                loss = criterion(logits, y[i : i + batch_size])
                loss.backward()
                optimizer.step()

        mlp.eval()
        accuracies = {}
        for t in tasks_seen:
            with torch.no_grad():
                logits = mlp(t.test_inputs)
                preds = logits.argmax(dim=1)
                acc = (preds == t.test_labels).float().mean().item()
            accuracies[t.task_id] = acc

        avg_acc = sum(accuracies.values()) / len(accuracies)
        results.append({
            "after_task": task.task_id,
            "accuracies": accuracies,
            "avg_accuracy": avg_acc,
        })

        acc_str = ", ".join(
            f"T{tid}={acc:.0%}" for tid, acc in sorted(accuracies.items())
        )
        print(f"  After Task {task.task_id}: {acc_str}  (avg={avg_acc:.0%})")

    return results


def plot_results(
    eam_results: list[dict],
    mlp_results: list[dict],
    training_history: list[dict],
    save_path: str = "synapse_results.png",
) -> None:
    """Generate comparison plot: training curve + EAM vs MLP continual learning."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # --- Subplot 1: Training curve ---
    ax1 = axes[0]
    epochs = [h["epoch"] for h in training_history]
    losses = [h["loss"] for h in training_history]
    accs = [h["accuracy"] for h in training_history]

    color_loss = "#d62728"
    color_acc = "#2ca02c"
    ax1.plot(epochs, losses, color=color_loss, linewidth=1.5, label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax1r = ax1.twinx()
    ax1r.plot(epochs, accs, color=color_acc, linewidth=1.5, label="Accuracy")
    ax1r.set_ylabel("Accuracy", color=color_acc)
    ax1r.tick_params(axis="y", labelcolor=color_acc)
    ax1r.set_ylim(0, 1.05)

    ax1.set_title("Phase 1: Controller Training (direct prototype matching)")

    # --- Subplot 2: EAM continual learning ---
    _plot_continual(axes[1], eam_results, "Phase 2: EAM — No Catastrophic Forgetting")

    # --- Subplot 3: MLP baseline ---
    _plot_continual(axes[2], mlp_results, "Phase 3: Standard MLP — Catastrophic Forgetting")

    # Summary annotation
    eam_final = eam_results[-1]["avg_accuracy"]
    mlp_final = mlp_results[-1]["avg_accuracy"]
    fig.text(
        0.5, 0.01,
        f"Final avg accuracy — EAM: {eam_final:.0%}  |  MLP: {mlp_final:.0%}",
        ha="center", fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_continual(ax: plt.Axes, results: list[dict], title: str) -> None:
    """Plot per-task accuracy lines over the continual learning stream."""
    all_task_ids = set()
    for r in results:
        all_task_ids.update(r["accuracies"].keys())
    all_task_ids = sorted(all_task_ids)

    x_labels = [f"After T{r['after_task']}" for r in results]
    x_pos = list(range(len(results)))

    colors = plt.cm.tab10.colors
    for i, tid in enumerate(all_task_ids):
        ys = []
        xs = []
        for j, r in enumerate(results):
            if tid in r["accuracies"]:
                ys.append(r["accuracies"][tid])
                xs.append(j)
        ax.plot(xs, ys, marker="o", color=colors[i % 10], linewidth=2, label=f"Task {tid}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
