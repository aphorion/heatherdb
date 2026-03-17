"""Differentiable Elastic Associative Memory layer.

PyTorch translation of heather_db's Elastic Associative Memory.
Read is differentiable (gradients flow to the query producer).
Write is non-differentiable (knowledge storage only).

Key Rust sources translated:
    read.rs::hopfield_iter   → EAMLayer.read()
    read.rs::activate (top-k) → torch.topk in both read and write
    write.rs::adaptive_write → EAMLayer.write()
    vec_ops.rs               → F.normalize, F.softmax, torch.mm
    location.rs              → addresses/counters/write_counts as buffers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EAMLayer(nn.Module):
    """Elastic Associative Memory as a differentiable PyTorch module.

    Stores knowledge in distributed counters across hard locations.
    Reads use Hopfield iterative dynamics with top-k activation.
    Writes update counters and adapt addresses via competitive learning.

    All state (addresses, counters, write_counts) is stored as buffers,
    NOT parameters — they are never updated by gradient descent.

    Gradients flow through the read operation: top-k values are
    differentiable w.r.t. the query (index selection is not, but the
    similarity values at those indices are). This is a straight-through
    pattern — the controller learns to produce queries with high
    similarity to the right locations.
    """

    def __init__(
        self,
        num_locations: int = 500,
        dim: int = 64,
        k: int = 20,
        beta: float = 5.0,
        t_max: int = 3,
        eta: float = 0.01,
    ):
        super().__init__()
        self.num_locations = num_locations
        self.dim = dim
        self.k = k
        self.beta = beta
        self.t_max = t_max
        self.eta = eta

        # Random unit-norm addresses (like Collection::new in collection.rs)
        addresses = torch.randn(num_locations, dim)
        addresses = F.normalize(addresses, dim=1)
        self.register_buffer("addresses", addresses)

        # Counters and write counts (like HardLocation fields in location.rs)
        self.register_buffer("counters", torch.zeros(num_locations, dim))
        self.register_buffer("write_counts", torch.zeros(num_locations))

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Differentiable Hopfield iterative read with top-k activation.

        Mirrors: heather_db/src/read.rs::hopfield_iter + activate

        Top-k selects the k most similar locations. Softmax over only
        those k locations gives sharp attention. Gradients flow through
        the similarity values (not through index selection).

        Args:
            query: [batch, dim] or [dim]
        Returns:
            Reconstructed vector, same shape as query.
        """
        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)

        # Compute patterns: unit_pattern() = normalize(counter / write_count)
        wc = self.write_counts.clamp(min=1e-12).unsqueeze(1)  # [L, 1]
        raw_patterns = self.counters / wc  # [L, dim]
        has_writes = self.write_counts > 0  # [L]
        patterns = F.normalize(raw_patterns, dim=1)  # [L, dim]
        # Zero out patterns for empty locations
        patterns = patterns * has_writes.float().unsqueeze(1)

        xi = F.normalize(query, dim=1)  # [B, dim]
        k = min(self.k, has_writes.sum().item())
        if k == 0:
            # No data written yet — return query unchanged
            if squeeze:
                xi = xi.squeeze(0)
            return xi

        for _ in range(self.t_max):
            # Cosine similarity (both unit-norm)
            sims = torch.mm(xi, self.addresses.t())  # [B, L]

            # Top-k selection (like read.rs::activate with min-heap)
            topk_sims, topk_idx = torch.topk(sims, k, dim=1)  # [B, k]

            # Softmax over only top-k (sharp attention)
            alpha = F.softmax(topk_sims * self.beta, dim=1)  # [B, k]

            # Gather patterns for top-k locations
            # topk_idx: [B, k] → expand to [B, k, dim]
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, self.dim)
            topk_patterns = patterns.unsqueeze(0).expand(xi.shape[0], -1, -1)
            topk_patterns = torch.gather(topk_patterns, 1, idx_expanded)  # [B, k, dim]

            # Weighted sum: [B, k, 1] * [B, k, dim] → sum → [B, dim]
            xi = (alpha.unsqueeze(-1) * topk_patterns).sum(dim=1)
            xi = F.normalize(xi, dim=1)

        if squeeze:
            xi = xi.squeeze(0)
        return xi

    @torch.no_grad()
    def write(self, vector: torch.Tensor) -> None:
        """Non-differentiable write to memory with top-k activation.

        Simplified from: heather_db/src/write.rs::adaptive_write
        Keeps: top-k activation, counter update, competitive learning
        Skips: conscience mechanism, novelty/overload splits

        Args:
            vector: [dim] or [batch, dim]
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        k = min(self.k, self.num_locations)

        for v in vector:
            v_norm = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)  # [dim]

            # Cosine similarities to all addresses
            sims = torch.mv(self.addresses, v_norm)  # [L]

            # Top-k activation (like read.rs::activate)
            topk_sims, topk_idx = torch.topk(sims, k)  # [k]

            # Softmax weights over top-k only
            weights = F.softmax(topk_sims * self.beta, dim=0)  # [k]

            # Counter update: c_j += w_j * v (only for top-k locations)
            for j, idx in enumerate(topk_idx):
                self.counters[idx] += weights[j] * v_norm
                self.write_counts[idx] += weights[j]

            # Competitive learning: winner address moves toward input
            winner = topk_idx[0]  # highest similarity
            diff = v_norm - self.addresses[winner]
            self.addresses[winner] += self.eta * diff
            self.addresses[winner] = F.normalize(
                self.addresses[winner], dim=0
            )

    @torch.no_grad()
    def clear(self) -> None:
        """Reset counters and write counts. Addresses preserved."""
        self.counters.zero_()
        self.write_counts.zero_()

    def num_written(self) -> int:
        """Number of locations with at least one write."""
        return (self.write_counts > 0).sum().item()
