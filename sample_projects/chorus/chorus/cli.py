"""Chorus CLI — interactive signal denoising through SDM."""

from __future__ import annotations

import os

import numpy as np

from .client import HeatherClient
from .signals import (
    generate_sine, generate_square, generate_composite, generate_chirp,
    add_noise, measure_snr,
)
from .engine import Chorus


# ── ASCII waveform chart ────────────────────────────────────────────

def ascii_waveform(signal: np.ndarray, width: int = 60, height: int = 8,
                   label: str = "") -> str:
    """Render a signal as an ASCII waveform."""
    if len(signal) == 0:
        return f"  {label}: (empty)"

    # Resample to width
    indices = np.linspace(0, len(signal) - 1, width).astype(int)
    samples = signal[indices]

    # Map to rows
    mn, mx = np.min(samples), np.max(samples)
    span = mx - mn
    if span < 1e-10:
        span = 1.0

    rows = []
    for r in range(height):
        row_val = mx - (r / (height - 1)) * span
        chars = []
        for s in samples:
            # Which row does this sample belong to?
            sample_row = int((mx - s) / span * (height - 1))
            if sample_row == r:
                chars.append("█")
            else:
                chars.append(" ")
        rows.append("".join(chars))

    # Add axis labels
    lines = []
    if label:
        lines.append(f"  {label}:")
    lines.append(f"  {mx:+.2f} ┤{''.join(rows[0])}")
    for r in range(1, height - 1):
        lines.append(f"        │{''.join(rows[r])}")
    lines.append(f"  {mn:+.2f} ┤{''.join(rows[-1])}")
    lines.append(f"        └{'─' * width}")
    return "\n".join(lines)


# ── State ────────────────────────────────────────────────────────

class State:
    def __init__(self):
        self.clean: np.ndarray | None = None
        self.noisy: np.ndarray | None = None
        self.denoised: np.ndarray | None = None
        self.snr_db: float = 10.0
        self.signal_name: str = ""
        self.trained: bool = False
        self.last_fidelities: list[float] = []


# ── Commands ─────────────────────────────────────────────────────

def cmd_sine(chorus: Chorus, state: State, args: list[str]):
    freq = float(args[0]) if len(args) > 0 else 440.0
    snr = float(args[1]) if len(args) > 1 else 10.0
    state.clean = generate_sine(freq, duration=0.5)
    state.snr_db = snr
    state.noisy = add_noise(state.clean, snr)
    state.signal_name = f"sine {freq}Hz"
    state.trained = False
    state.denoised = None
    actual_snr = measure_snr(state.clean, state.noisy)
    print(f"Generated: {state.signal_name}, {len(state.clean)} samples")
    print(f"Added noise: target {snr:.0f} dB, actual {actual_snr:.1f} dB")


def cmd_square(chorus: Chorus, state: State, args: list[str]):
    freq = float(args[0]) if len(args) > 0 else 440.0
    snr = float(args[1]) if len(args) > 1 else 10.0
    state.clean = generate_square(freq, duration=0.5)
    state.snr_db = snr
    state.noisy = add_noise(state.clean, snr)
    state.signal_name = f"square {freq}Hz"
    state.trained = False
    state.denoised = None
    actual_snr = measure_snr(state.clean, state.noisy)
    print(f"Generated: {state.signal_name}, {len(state.clean)} samples")
    print(f"Added noise: target {snr:.0f} dB, actual {actual_snr:.1f} dB")


def cmd_composite(chorus: Chorus, state: State, args: list[str]):
    snr = float(args[0]) if len(args) > 0 else 10.0
    state.clean = generate_composite()
    state.snr_db = snr
    state.noisy = add_noise(state.clean, snr)
    state.signal_name = "composite (220+440+880 Hz)"
    state.trained = False
    state.denoised = None
    actual_snr = measure_snr(state.clean, state.noisy)
    print(f"Generated: {state.signal_name}, {len(state.clean)} samples")
    print(f"Added noise: target {snr:.0f} dB, actual {actual_snr:.1f} dB")


def cmd_chirp(chorus: Chorus, state: State, args: list[str]):
    snr = float(args[0]) if len(args) > 0 else 10.0
    state.clean = generate_chirp()
    state.snr_db = snr
    state.noisy = add_noise(state.clean, snr)
    state.signal_name = "chirp (200→800 Hz)"
    state.trained = False
    state.denoised = None
    actual_snr = measure_snr(state.clean, state.noisy)
    print(f"Generated: {state.signal_name}, {len(state.clean)} samples")
    print(f"Added noise: target {snr:.0f} dB, actual {actual_snr:.1f} dB")


def cmd_train(chorus: Chorus, state: State, args: list[str]):
    if state.clean is None:
        print("No signal loaded. Use: sine, square, composite, or chirp")
        return
    n = int(args[0]) if len(args) > 0 else 10
    print(f"Ingesting {n} noisy copies at {state.snr_db:.0f} dB SNR...")
    count = chorus.ingest_copies(state.clean, n, state.snr_db)
    state.trained = True
    print(f"Wrote {count} window vectors to SDM (total: {chorus.total_written})")


def cmd_denoise(chorus: Chorus, state: State, args: list[str]):
    if state.clean is None or state.noisy is None:
        print("No signal loaded.")
        return
    if not state.trained:
        print("Not trained yet. Use: train [n_copies]")
        return

    print("Denoising...")
    result = chorus.denoise_signal(state.clean, state.noisy)
    state.denoised = result.denoised
    state.last_fidelities = result.fidelities

    print(f"\n  Input SNR:  {result.input_snr:+.1f} dB")
    print(f"  Output SNR: {result.output_snr:+.1f} dB")
    print(f"  Improvement: {result.improvement_db:+.1f} dB")
    print(f"  Mean fidelity: {result.mean_fidelity:.4f}")

    # Quick quality bar
    imp = result.improvement_db
    if imp > 5:
        quality = "Excellent"
    elif imp > 2:
        quality = "Good"
    elif imp > 0:
        quality = "Marginal"
    else:
        quality = "No improvement"
    bar_len = max(0, min(30, int(imp)))
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  Quality: [{bar}] {quality}")
    print()


def cmd_chart(chorus: Chorus, state: State, args: list[str]):
    if state.clean is None:
        print("No signal loaded.")
        return

    print()
    print(ascii_waveform(state.clean, label="Clean"))
    print()
    if state.noisy is not None:
        print(ascii_waveform(state.noisy, label="Noisy"))
        print()
    if state.denoised is not None:
        print(ascii_waveform(state.denoised, label="Denoised"))
        print()
        # Show error
        error = state.denoised - state.clean[:len(state.denoised)]
        print(ascii_waveform(error, label="Error (denoised - clean)"))
        print()


def cmd_fidelity(chorus: Chorus, state: State, args: list[str]):
    if not state.last_fidelities:
        print("No fidelity data. Run denoise first.")
        return

    fids = state.last_fidelities
    print(f"\nPer-window fidelity ({len(fids)} windows):")
    print(f"  Min: {min(fids):.4f}  Max: {max(fids):.4f}  "
          f"Mean: {np.mean(fids):.4f}  Std: {np.std(fids):.4f}\n")

    # ASCII sparkline
    width = min(60, len(fids))
    indices = np.linspace(0, len(fids) - 1, width).astype(int)
    sampled = [fids[i] for i in indices]

    mn, mx = min(sampled), max(sampled)
    blocks = " ▁▂▃▄▅▆▇█"

    chars = []
    for f in sampled:
        if mx - mn < 1e-6:
            idx = 8
        else:
            idx = int((f - mn) / (mx - mn) * 8)
        chars.append(blocks[idx])
    print(f"  {''.join(chars)}")
    print(f"  {mn:.3f}{' ' * (width - 10)}{mx:.3f}")
    print()


def cmd_sweep(chorus: Chorus, state: State, args: list[str]):
    if state.clean is None:
        print("No signal loaded.")
        return

    print(f"Running copy count sweep at {state.snr_db:.0f} dB SNR...")
    print("(copies accumulate in SDM — showing incremental effect)\n")

    result = chorus.sweep(state.clean, state.snr_db)

    print(f"  {'Copies':>8}  {'In SNR':>8}  {'Out SNR':>8}  {'Gain':>8}  Bar")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")
    for n, in_snr, out_snr, gain in result.entries:
        bar_len = max(0, min(20, int(gain)))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {n:>8}  {in_snr:>+8.1f}  {out_snr:>+8.1f}  {gain:>+8.1f}  {bar}")
    print()

    state.trained = True


def cmd_test(chorus: Chorus, state: State, args: list[str]):
    """Test superposition on a single 128-sample segment (no windowing)."""
    n = int(args[0]) if len(args) > 0 else 20
    snr = float(args[1]) if len(args) > 1 else 10.0

    # Generate one clean segment (128 samples of sine)
    segment = generate_sine(440.0, duration=128/8000, sample_rate=8000)
    if len(segment) < 128:
        segment = np.pad(segment, (0, 128 - len(segment)))
    segment = segment[:128]

    print(f"Testing: {n} noisy copies of single 128-sample sine segment")
    print(f"Noise: {snr:.0f} dB SNR\n")

    result = chorus.test_single_window(segment, n, snr)

    print(f"  Input SNR:         {result['input_snr']:+.1f} dB")
    print(f"  SDM (fast):        {result['output_snr_fast']:+.1f} dB  "
          f"({result['improvement_fast']:+.1f} dB)  "
          f"fidelity={result['fidelity_fast']:.4f}")
    print(f"  SDM (iterative):   {result['output_snr_iter']:+.1f} dB  "
          f"({result['improvement_iter']:+.1f} dB)  "
          f"fidelity={result['fidelity_iter']:.4f}")
    print(f"  Direct average:    {result['direct_avg_snr']:+.1f} dB  "
          f"({result['direct_avg_improvement']:+.1f} dB)")
    print()
    print("  Direct average = numpy mean of N noisy copies (theoretical best)")
    print("  If SDM matches direct average, superposition is working.")
    print()


def cmd_stats(chorus: Chorus, state: State, args: list[str]):
    try:
        stats = chorus.heather.stats()
        print(f"\nHeatherDB:")
        print(f"  Hard locations: {stats.get('hard_location_count', '?')}")
        print(f"  Dimension: {stats.get('dimension', '?')}")
        print(f"  Writes: {stats.get('total_writes', '?')}")
    except Exception as e:
        print(f"  HeatherDB error: {e}")

    print(f"\nChorus:")
    print(f"  Total vectors written: {chorus.total_written}")
    if state.signal_name:
        print(f"  Active signal: {state.signal_name}")
        print(f"  Noise level: {state.snr_db:.0f} dB")
    print()


# ── Main ─────────────────────────────────────────────────────────

def run_cli():
    url = os.getenv("HEATHER_URL", "http://localhost:6380")

    print("=" * 60)
    print("  CHORUS — Signal Denoising Through SDM Superposition")
    print("=" * 60)
    print()

    heather = HeatherClient(url)
    if not heather.health():
        print(f"WARNING: HeatherDB not reachable at {url}")
        print("Start it with: cargo run --release -p heather_server "
              "-- --dimension 128 --port 6380")
        print()

    chorus = Chorus(heather)
    state = State()

    commands = {
        "sine": cmd_sine,
        "square": cmd_square,
        "composite": cmd_composite,
        "chirp": cmd_chirp,
        "train": cmd_train,
        "denoise": cmd_denoise,
        "chart": cmd_chart,
        "fidelity": cmd_fidelity,
        "sweep": cmd_sweep,
        "test": cmd_test,
        "stats": cmd_stats,
    }

    print("Commands:")
    print("  sine [freq] [noise_db]    — generate sine wave (default 440Hz, 10dB)")
    print("  square [freq] [noise_db]  — generate square wave")
    print("  composite [noise_db]      — sum of 3 sines (220+440+880)")
    print("  chirp [noise_db]          — frequency sweep (200→800 Hz)")
    print("  train [n_copies]          — ingest N noisy copies into EAM (default 10)")
    print("  denoise                   — denoise active signal, show SNR improvement")
    print("  chart                     — ASCII waveforms: clean / noisy / denoised")
    print("  fidelity                  — per-window fidelity sparkline")
    print("  sweep                     — copy count sweep (1→50), SNR vs N")
    print("  test [n_copies] [snr_db]  — single-window test (isolates superposition)")
    print("  stats                     — show stats")
    print("  quit                      — exit")
    print()

    while True:
        try:
            line = input("chorus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue
        if line in ("quit", "exit", "q"):
            print("Bye!")
            break

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in commands:
            try:
                commands[cmd](chorus, state, args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}")
