"""Neural controller that learns HOW to use memory.

A small MLP that projects raw input space into memory space.
This is the only component updated by gradient descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    """Encoder that maps inputs to unit-norm memory-space vectors.

    Used for both writes (producing what to store) and reads (producing
    what to query). Single shared encoder ensures the two spaces align.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        memory_dim: int = 64,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to normalized memory-space vector.

        Args:
            x: [batch, input_dim] or [input_dim]
        Returns:
            Unit-normalized vector in memory space.
        """
        h = self.encoder(x)
        return F.normalize(h, dim=-1)
