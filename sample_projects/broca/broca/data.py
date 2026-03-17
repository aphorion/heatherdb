"""TinyShakespeare dataset for chunk-level language modeling."""

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class CharVocab:
    """Character-level vocabulary built from corpus text."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in indices)


class ChunkDataset(Dataset):
    """Sliding-window dataset: (context, target_chunk) pairs.

    Each sample:
        context: [context_length] characters of input
        target:  [chunk_size] characters immediately following context
    """

    def __init__(
        self,
        text: str,
        vocab: CharVocab,
        context_length: int = 128,
        chunk_size: int = 32,
    ):
        self.context_length = context_length
        self.chunk_size = chunk_size
        self.data = torch.tensor(vocab.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.context_length - self.chunk_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.data[idx : idx + self.context_length]
        target = self.data[
            idx + self.context_length : idx + self.context_length + self.chunk_size
        ]
        return context, target


def download_shakespeare(data_dir: str = "./data") -> str:
    """Download TinyShakespeare if not cached. Return the text."""
    path = Path(data_dir) / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading TinyShakespeare to {path}...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, str(path))
    return path.read_text()
