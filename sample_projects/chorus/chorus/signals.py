"""Signal generation, noise, and windowing utilities."""

from __future__ import annotations

import numpy as np


def generate_sine(freq: float = 440.0, duration: float = 0.5,
                  sample_rate: int = 8000) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * freq * t)


def generate_square(freq: float = 440.0, duration: float = 0.5,
                    sample_rate: int = 8000) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.sign(np.sin(2 * np.pi * freq * t))


def generate_composite(freqs: list[float] | None = None,
                       amplitudes: list[float] | None = None,
                       duration: float = 0.5,
                       sample_rate: int = 8000) -> np.ndarray:
    if freqs is None:
        freqs = [220.0, 440.0, 880.0]
    if amplitudes is None:
        amplitudes = [1.0, 0.5, 0.25]
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    for f, a in zip(freqs, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)
    # Normalize to [-1, 1]
    mx = np.max(np.abs(signal))
    if mx > 0:
        signal /= mx
    return signal


def generate_chirp(f0: float = 200.0, f1: float = 800.0,
                   duration: float = 0.5,
                   sample_rate: int = 8000) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    return np.sin(phase)


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at a target SNR (in dB)."""
    sig_power = np.mean(signal**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure SNR in dB between clean signal and noisy/denoised version."""
    noise = noisy - clean
    sig_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-15:
        return 100.0  # effectively perfect
    return 10 * np.log10(sig_power / noise_power)


def window_signal(signal: np.ndarray, window_size: int = 64,
                  stride: int = 32) -> list[np.ndarray]:
    """Split signal into overlapping windows."""
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size].copy())
    return windows


def reconstruct_from_windows(windows: list[np.ndarray],
                             window_size: int = 64,
                             stride: int = 32,
                             total_length: int | None = None) -> np.ndarray:
    """Overlap-add reconstruction from windows."""
    if not windows:
        return np.array([])
    if total_length is None:
        total_length = (len(windows) - 1) * stride + window_size
    result = np.zeros(total_length)
    counts = np.zeros(total_length)
    for i, w in enumerate(windows):
        start = i * stride
        end = start + window_size
        if end > total_length:
            break
        result[start:end] += w
        counts[start:end] += 1
    # Average overlapping regions
    mask = counts > 0
    result[mask] /= counts[mask]
    return result
