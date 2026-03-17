"""Character-level language model: GRU encoder + EAM + GRU decoder.

Think then Speak architecture:
  Encoder GRU compresses context into a thought vector (one read).
  EAM pattern-completes the thought (one disk access).
  Decoder GRU renders the thought into a chunk of characters (pure RAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eam import EAMLayer


class CharLM(nn.Module):
    """Character-level language model with chunk-level generation.

    Components:
        embedding:    vocab_size → embed_dim lookup (shared encoder/decoder)
        encoder_gru:  GRU(embed_dim, memory_dim), context → thought
        decoder_gru:  GRU(embed_dim, memory_dim), thought → chunk
        output:       Linear(memory_dim, vocab_size), hidden → logits
        memory:       EAMLayer, associative knowledge store
    """

    def __init__(
        self,
        vocab_size: int = 65,
        embed_dim: int = 64,
        memory_dim: int = 128,
        context_length: int = 128,
        chunk_size: int = 32,
        num_locations: int = 2000,
        k: int = 20,
        beta: float = 5.0,
        t_max: int = 3,
        eta: float = 0.001,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.context_length = context_length
        self.chunk_size = chunk_size

        # Shared character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoder: context sequence → thought vector
        self.encoder_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=memory_dim,
            num_layers=1,
            batch_first=True,
        )

        # Decoder: thought vector → chunk of characters
        self.decoder_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=memory_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(memory_dim, vocab_size)

        # EAM: associative knowledge store (buffers, not parameters)
        self.memory = EAMLayer(
            num_locations=num_locations,
            dim=memory_dim,
            k=k,
            beta=beta,
            t_max=t_max,
            eta=eta,
        )

    def encode(self, char_indices: torch.Tensor) -> torch.Tensor:
        """Encode character sequence to unit-norm thought vector.

        Args:
            char_indices: [batch, seq_len] integer character indices
        Returns:
            [batch, memory_dim] unit-normalized thought vectors
        """
        emb = self.embedding(char_indices)      # [B, T, embed_dim]
        _, hidden = self.encoder_gru(emb)       # hidden: [1, B, memory_dim]
        thought = hidden.squeeze(0)             # [B, memory_dim]
        return F.normalize(thought, dim=-1)

    def decode_chunk(
        self,
        thought: torch.Tensor,
        seed_chars: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced chunk decoding (training).

        Processes the full chunk in one GRU pass since all inputs are known.

        Args:
            thought: [B, memory_dim] thought vector → decoder initial hidden
            seed_chars: [B] last char of context (first decoder input)
            target_chunk: [B, chunk_size] ground truth target chars
        Returns:
            logits: [B, chunk_size, vocab_size]
        """
        # Decoder input: [seed, t_0, t_1, ..., t_{N-2}]
        # Predicts:       [t_0,  t_1, t_2, ..., t_{N-1}]
        decoder_input = torch.cat(
            [seed_chars.unsqueeze(1), target_chunk[:, :-1]], dim=1
        )  # [B, chunk_size]

        emb = self.embedding(decoder_input)                        # [B, chunk_size, embed_dim]
        output, _ = self.decoder_gru(emb, thought.unsqueeze(0))   # [B, chunk_size, memory_dim]
        return self.output(output)                                 # [B, chunk_size, vocab_size]

    @torch.no_grad()
    def generate_chunk(
        self,
        thought: torch.Tensor,
        seed_char: torch.Tensor,
        temperature: float = 0.8,
    ) -> list[int]:
        """Autoregressive chunk generation (inference).

        One thought → one chunk. The decoder runs in pure RAM.

        Args:
            thought: [1, memory_dim] thought vector
            seed_char: [1] seed character (last char of context)
            temperature: sampling temperature
        Returns:
            List of chunk_size character indices
        """
        hidden = thought.unsqueeze(0)   # [1, 1, memory_dim]
        tokens = []
        current = seed_char             # [1]

        for _ in range(self.chunk_size):
            emb = self.embedding(current).unsqueeze(1)          # [1, 1, embed_dim]
            out, hidden = self.decoder_gru(emb, hidden)         # [1, 1, memory_dim]
            logits = self.output(out.squeeze(1)) / temperature  # [1, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)
            tokens.append(next_token.item())
            current = next_token

        return tokens

    def forward(
        self,
        char_indices: torch.Tensor,
        target_chunk: torch.Tensor,
        use_memory: bool = False,
    ) -> torch.Tensor:
        """Full forward pass: encode → (optional EAM) → decode chunk.

        Args:
            char_indices: [B, context_length] context characters
            target_chunk: [B, chunk_size] target characters for teacher forcing
            use_memory: route through EAM before decoding
        Returns:
            logits: [B, chunk_size, vocab_size]
        """
        thought = self.encode(char_indices)

        if use_memory and self.memory.num_written() > 0:
            thought = self.memory.read(thought)

        seed_chars = char_indices[:, -1]
        return self.decode_chunk(thought, seed_chars, target_chunk)

    def freeze_controller(self):
        """Freeze all learnable parameters for Phase 2 EAM population."""
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        for param in self.encoder_gru.parameters():
            param.requires_grad_(False)
        for param in self.decoder_gru.parameters():
            param.requires_grad_(False)
        for param in self.output.parameters():
            param.requires_grad_(False)
