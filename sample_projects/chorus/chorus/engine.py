"""Chorus engine — signal denoising through SDM superposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .client import HeatherClient
from .signals import (
    add_noise, measure_snr, window_signal, reconstruct_from_windows,
)
from .encoding import encode_window, decode_window, cosine_similarity, WINDOW_SIZE


STRIDE = 64  # 50% overlap for 128-sample windows


@dataclass
class DenoiseResult:
    """Result of denoising a signal."""
    denoised: np.ndarray
    input_snr: float
    output_snr: float
    improvement_db: float
    fidelities: list[float]
    mean_fidelity: float


@dataclass
class SweepResult:
    """Result of sweeping over copy counts."""
    snr_db: float
    entries: list[tuple[int, float, float, float]]


@dataclass
class NoiseSweepResult:
    """Result of sweeping over noise levels."""
    n_copies: int
    entries: list[tuple[float, float, float]]


class Chorus:
    """Signal denoising through SDM superposition."""

    def __init__(self, heather: HeatherClient):
        self.heather = heather
        self.total_written = 0

    def ingest_copies(self, clean: np.ndarray, n_copies: int,
                      snr_db: float) -> int:
        """Generate N noisy copies, encode with position binding, write to SDM."""
        count = 0
        for _ in range(n_copies):
            noisy = add_noise(clean, snr_db)
            windows = window_signal(noisy, WINDOW_SIZE, STRIDE)
            vecs = [encode_window(w, position=i) for i, w in enumerate(windows)]
            if vecs:
                self.heather.write(vecs)
                count += len(vecs)
        self.total_written += count
        return count

    def denoise_window(self, noisy_window: np.ndarray,
                       position: int = 0) -> tuple[np.ndarray, float]:
        """Denoise a single window via EAM reconstruction."""
        original_norm = float(np.linalg.norm(noisy_window))
        vec = encode_window(noisy_window, position=position)
        reconstructed = self.heather.read(vec, strategy="fast")
        fidelity = cosine_similarity(vec, reconstructed)
        denoised = decode_window(reconstructed, position=position,
                                 target_norm=original_norm)
        return denoised, fidelity

    def denoise_signal(self, clean: np.ndarray,
                       noisy: np.ndarray) -> DenoiseResult:
        """Denoise an entire signal."""
        input_snr = measure_snr(clean, noisy)
        windows = window_signal(noisy, WINDOW_SIZE, STRIDE)

        denoised_windows = []
        fidelities = []
        for i, w in enumerate(windows):
            dw, fid = self.denoise_window(w, position=i)
            denoised_windows.append(dw)
            fidelities.append(fid)

        denoised = reconstruct_from_windows(
            denoised_windows, WINDOW_SIZE, STRIDE, len(clean)
        )

        if len(denoised) < len(clean):
            denoised = np.pad(denoised, (0, len(clean) - len(denoised)))
        else:
            denoised = denoised[:len(clean)]

        output_snr = measure_snr(clean, denoised)

        return DenoiseResult(
            denoised=denoised,
            input_snr=input_snr,
            output_snr=output_snr,
            improvement_db=output_snr - input_snr,
            fidelities=fidelities,
            mean_fidelity=float(np.mean(fidelities)) if fidelities else 0.0,
        )

    def sweep(self, clean: np.ndarray, snr_db: float,
              copy_counts: list[int] | None = None) -> SweepResult:
        """Sweep over copy counts to show SNR improvement vs N."""
        if copy_counts is None:
            copy_counts = [1, 2, 5, 10, 20, 50]

        noisy = add_noise(clean, snr_db)
        entries = []
        prev_count = 0

        for n in copy_counts:
            delta = n - prev_count
            if delta > 0:
                self.ingest_copies(clean, delta, snr_db)
            prev_count = n

            result = self.denoise_signal(clean, noisy)
            entries.append((n, result.input_snr, result.output_snr,
                            result.improvement_db))

        return SweepResult(snr_db=snr_db, entries=entries)

    def noise_sweep(self, clean: np.ndarray, n_copies: int,
                    snr_levels: list[float] | None = None) -> NoiseSweepResult:
        """Fixed copies, varying noise — find the breaking point."""
        if snr_levels is None:
            snr_levels = [0.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]

        self.ingest_copies(clean, n_copies, 10.0)

        entries = []
        for snr in snr_levels:
            noisy = add_noise(clean, snr)
            result = self.denoise_signal(clean, noisy)
            entries.append((result.input_snr, result.output_snr,
                            result.improvement_db))

        return NoiseSweepResult(n_copies=n_copies, entries=entries)

    def test_single_window(self, clean_segment: np.ndarray, n_copies: int,
                           snr_db: float) -> dict:
        """Pure test: write N noisy copies of ONE 128-sample segment."""
        assert len(clean_segment) == WINDOW_SIZE

        vecs = []
        for _ in range(n_copies):
            noisy = add_noise(clean_segment, snr_db)
            vecs.append(encode_window(noisy, position=0))
        self.heather.write(vecs)
        self.total_written += len(vecs)

        query_noisy = add_noise(clean_segment, snr_db)
        query_norm = float(np.linalg.norm(query_noisy))
        query_vec = encode_window(query_noisy, position=0)

        reconstructed = self.heather.read(query_vec, strategy="fast")
        fidelity = cosine_similarity(query_vec, reconstructed)
        denoised = decode_window(reconstructed, position=0,
                                 target_norm=query_norm)

        reconstructed_iter = self.heather.read(query_vec, strategy="iterative")
        fidelity_iter = cosine_similarity(query_vec, reconstructed_iter)
        denoised_iter = decode_window(reconstructed_iter, position=0,
                                      target_norm=query_norm)

        input_snr = measure_snr(clean_segment, query_noisy)
        output_snr_fast = measure_snr(clean_segment, denoised)
        output_snr_iter = measure_snr(clean_segment, denoised_iter)

        avg_noisy = np.mean([add_noise(clean_segment, snr_db)
                             for _ in range(n_copies)], axis=0)
        avg_snr = measure_snr(clean_segment, avg_noisy)

        return {
            "n_copies": n_copies,
            "snr_db": snr_db,
            "input_snr": input_snr,
            "output_snr_fast": output_snr_fast,
            "output_snr_iter": output_snr_iter,
            "improvement_fast": output_snr_fast - input_snr,
            "improvement_iter": output_snr_iter - input_snr,
            "fidelity_fast": fidelity,
            "fidelity_iter": fidelity_iter,
            "direct_avg_snr": avg_snr,
            "direct_avg_improvement": avg_snr - input_snr,
        }
