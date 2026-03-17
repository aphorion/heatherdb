"""EAMClassifier: Controller + Elastic Associative Memory + prototypes.

The full model: controller encodes inputs, EAM stores/retrieves knowledge,
prototype matching produces classification logits.

Gradient flow:
    loss → logits → reconstruction → EAM.read → query → controller weights
    (EAM buffers participate in forward pass but receive no gradients)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eam import EAMLayer
from .controller import Controller


class EAMClassifier(nn.Module):
    """Neural controller + EAM memory + prototype-based classifier.

    The controller (nn.Parameters) learns HOW to encode for memory.
    The EAM (register_buffers) stores WHAT it knows.
    Classification uses prototype matching — no fixed head, supports
    unlimited new classes.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        memory_dim: int = 64,
        num_locations: int = 500,
        k: int = 20,
        beta: float = 5.0,
        t_max: int = 3,
        eta: float = 0.01,
        logit_scale: float = 10.0,
    ):
        super().__init__()
        self.controller = Controller(input_dim, hidden_dim, memory_dim)
        self.memory = EAMLayer(num_locations, memory_dim, k, beta, t_max, eta)
        self.logit_scale = logit_scale

        # Prototypes: class label (int) → mean encoded vector
        self._prototypes: dict[int, torch.Tensor] = {}
        self._prototype_counts: dict[int, int] = {}

        # Stored exemplar data for re-encoding during training
        self._exemplar_inputs: dict[int, torch.Tensor] = {}

    @property
    def class_labels(self) -> list[int]:
        return sorted(self._prototypes.keys())

    @torch.no_grad()
    def write_exemplars(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Encode inputs and write to EAM. Update class prototypes.

        Args:
            inputs: [N, input_dim]
            labels: [N] integer class labels
        """
        encoded = self.controller(inputs)  # [N, memory_dim]
        self.memory.write(encoded)

        # Update prototypes and store raw inputs for re-encoding
        for label_val in labels.unique():
            mask = labels == label_val
            class_vecs = encoded[mask]
            batch_mean = class_vecs.mean(dim=0)
            batch_n = class_vecs.shape[0]
            key = label_val.item()

            if key in self._prototypes:
                old_mean = self._prototypes[key]
                old_n = self._prototype_counts[key]
                new_n = old_n + batch_n
                self._prototypes[key] = (old_mean * old_n + batch_mean * batch_n) / new_n
                self._prototype_counts[key] = new_n
                self._exemplar_inputs[key] = torch.cat(
                    [self._exemplar_inputs[key], inputs[mask]]
                )
            else:
                self._prototypes[key] = batch_mean.clone()
                self._prototype_counts[key] = batch_n
                self._exemplar_inputs[key] = inputs[mask].clone()

    @torch.no_grad()
    def refresh_prototypes(self) -> None:
        """Recompute prototypes from stored exemplar inputs using current controller.

        Called during training to keep prototypes aligned with the
        controller's evolving representation.
        """
        for key, raw_inputs in self._exemplar_inputs.items():
            encoded = self.controller(raw_inputs)
            self._prototypes[key] = encoded.mean(dim=0)

    def _build_proto_matrix(self) -> torch.Tensor:
        """Stack prototypes into [C, memory_dim] matrix."""
        return torch.stack(
            [F.normalize(self._prototypes[k], dim=0) for k in self.class_labels]
        )

    def forward(self, x: torch.Tensor, use_memory: bool = True) -> torch.Tensor:
        """Classify via prototype matching, optionally through EAM.

        When use_memory=True (default): encode → EAM read → match prototypes
        When use_memory=False: encode → match prototypes directly
            (used during Phase 1 training to learn encoding strategy)

        Args:
            x: [batch, input_dim]
            use_memory: whether to route through EAM read
        Returns:
            logits: [batch, num_classes]
        """
        encoded = self.controller(x)  # [B, memory_dim] — differentiable

        if use_memory and self.memory.num_written() > 0:
            encoded = self.memory.read(encoded)  # [B, memory_dim] — differentiable

        proto_matrix = self._build_proto_matrix()  # [C, memory_dim]
        logits = torch.mm(encoded, proto_matrix.t())  # [B, C]
        return logits * self.logit_scale

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Returns: [batch] integer labels
        """
        logits = self.forward(x)
        labels = self.class_labels
        pred_indices = logits.argmax(dim=1)
        return torch.tensor([labels[i] for i in pred_indices])

    def clear_memory(self) -> None:
        """Reset EAM contents and prototypes."""
        self.memory.clear()
        self._prototypes.clear()
        self._prototype_counts.clear()
        self._exemplar_inputs.clear()
