"""MNIST classifier: encoder + decoder in memory, EAM in HeatherDB.

The encoder (controller neural net) maps raw 784-pixel images to 64-d
unit-norm memory vectors. The decoder (class prototypes) maps memory
vectors back to digit labels via cosine similarity. The associative
memory (EAM) lives server-side in HeatherDB.

Flow: image -> encode -> HeatherDB /read -> cosine match prototypes -> label
"""

import json
from pathlib import Path

import numpy as np
import torch

from .client import HeatherClient
from .controller import Controller


class MNISTClassifier:
    """Classifies MNIST digits via HeatherDB.

    Keeps the encoder (controller) and decoder (prototypes) in local
    memory. The heavy lifting — associative pattern completion via
    Hopfield dynamics — happens server-side in HeatherDB.
    """

    def __init__(
        self,
        export_path: str,
        heather: HeatherClient,
        device: str = "cpu",
    ):
        self.heather = heather
        self.device = torch.device(device)

        export_path = Path(export_path)
        with open(export_path) as f:
            export = json.load(f)

        # Load prototypes (decoder)
        self.prototypes: dict[int, np.ndarray] = {}
        for label_str, vec in export["prototypes"].items():
            self.prototypes[int(label_str)] = np.array(vec, dtype=np.float64)

        self.labels = sorted(self.prototypes.keys())
        self._proto_matrix = np.stack([self.prototypes[l] for l in self.labels])

        # Load controller (encoder)
        config = export["config"]
        ctrl_name = export.get("controller_state")
        if ctrl_name:
            ctrl_path = export_path.parent / ctrl_name
        else:
            ctrl_path = export_path.with_name(export_path.stem + "_controller.pt")

        self.controller = Controller(
            input_dim=784,
            hidden_dim=128,
            memory_dim=config["d"],
        )
        self.controller.load_state_dict(torch.load(ctrl_path, map_location=self.device, weights_only=True))
        self.controller.eval()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode raw 784-d images to unit-norm memory vectors."""
        with torch.no_grad():
            return self.controller(images.to(self.device))

    def classify(self, image: torch.Tensor) -> tuple[int, float]:
        """Classify a single image through HeatherDB.

        Returns (predicted_label, confidence).
        """
        encoded = self.encode(image.unsqueeze(0) if image.dim() == 1 else image)
        query = encoded[0].cpu().numpy().astype(np.float64).tolist()

        # EAM read via HeatherDB
        reconstruction = np.array(self.heather.read(query), dtype=np.float64)

        # Cosine similarity against prototypes (decoder)
        norm = np.linalg.norm(reconstruction)
        if norm < 1e-12:
            return self.labels[0], 0.0

        reconstruction = reconstruction / norm
        similarities = self._proto_matrix @ reconstruction

        best_idx = int(np.argmax(similarities))
        return self.labels[best_idx], float(similarities[best_idx])

    def classify_batch(
        self, images: torch.Tensor
    ) -> list[tuple[int, float]]:
        """Classify a batch of images. Each query goes through HeatherDB."""
        encoded = self.encode(images)
        results = []
        for i in range(encoded.shape[0]):
            query = encoded[i].cpu().numpy().astype(np.float64).tolist()
            reconstruction = np.array(self.heather.read(query), dtype=np.float64)

            norm = np.linalg.norm(reconstruction)
            if norm < 1e-12:
                results.append((self.labels[0], 0.0))
                continue

            reconstruction = reconstruction / norm
            similarities = self._proto_matrix @ reconstruction
            best_idx = int(np.argmax(similarities))
            results.append((self.labels[best_idx], float(similarities[best_idx])))

        return results
