"""Synthetic classification tasks as Gaussian clusters.

Each task introduces new classes (non-overlapping). Cluster centers are
random unit vectors in high-dimensional space — nearly orthogonal by
concentration of measure, so 15+ classes are easily separable.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Task:
    task_id: int
    class_labels: list[int]
    exemplars: torch.Tensor       # [classes*exemplars_per_class, input_dim]
    exemplar_labels: torch.Tensor
    train_inputs: torch.Tensor    # [classes*train_per_class, input_dim]
    train_labels: torch.Tensor
    test_inputs: torch.Tensor     # [classes*test_per_class, input_dim]
    test_labels: torch.Tensor


class TaskGenerator:
    """Generate synthetic Gaussian cluster classification tasks."""

    def __init__(
        self,
        input_dim: int = 128,
        num_tasks: int = 5,
        classes_per_task: int = 3,
        exemplars_per_class: int = 50,
        train_per_class: int = 200,
        test_per_class: int = 100,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.exemplars_per_class = exemplars_per_class
        self.train_per_class = train_per_class
        self.test_per_class = test_per_class
        self.noise_std = noise_std
        self.seed = seed

    def generate(self) -> list[Task]:
        gen = torch.Generator().manual_seed(self.seed)
        tasks = []
        label_counter = 0

        for task_id in range(self.num_tasks):
            class_labels = []
            all_exemplars, all_ex_labels = [], []
            all_train, all_train_labels = [], []
            all_test, all_test_labels = [], []

            for _ in range(self.classes_per_task):
                label = label_counter
                label_counter += 1
                class_labels.append(label)

                # Random unit-vector center
                center = torch.randn(self.input_dim, generator=gen)
                center = F.normalize(center, dim=0)

                # Generate noisy samples around center
                for split, n in [
                    (all_exemplars, self.exemplars_per_class),
                    (all_train, self.train_per_class),
                    (all_test, self.test_per_class),
                ]:
                    samples = center.unsqueeze(0) + torch.randn(
                        n, self.input_dim, generator=gen
                    ) * self.noise_std
                    samples = F.normalize(samples, dim=1)
                    split.append(samples)

                n_ex = self.exemplars_per_class
                n_tr = self.train_per_class
                n_te = self.test_per_class
                all_ex_labels.append(torch.full((n_ex,), label, dtype=torch.long))
                all_train_labels.append(torch.full((n_tr,), label, dtype=torch.long))
                all_test_labels.append(torch.full((n_te,), label, dtype=torch.long))

            tasks.append(Task(
                task_id=task_id,
                class_labels=class_labels,
                exemplars=torch.cat(all_exemplars),
                exemplar_labels=torch.cat(all_ex_labels),
                train_inputs=torch.cat(all_train),
                train_labels=torch.cat(all_train_labels),
                test_inputs=torch.cat(all_test),
                test_labels=torch.cat(all_test_labels),
            ))

        return tasks
