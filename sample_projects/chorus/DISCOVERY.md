# Discovery: Signal Denoising Through SDM Superposition

**Date:** 2026-02-13
**Project:** Chorus (HeatherDB sample project)
**Finding:** SDM superposition naturally denoises signals — store N noisy copies, noise cancels while signal reinforces. More copies = cleaner reconstruction. A vector DB gets zero benefit from multiple copies.

## Setup

- Generate a clean signal (sine wave, 440Hz, 128-sample windows)
- Create N noisy copies (additive Gaussian noise at 10 dB SNR)
- Encode each window as a unit vector (signal IS the vector, 128 dims)
- **Position binding**: multiply by a random ±1 key per window position (holographic binding)
- Write all noisy copies to SDM
- Query with a new noisy window → EAM reconstruction = denoised signal

## Key Results

### Copy Count Sweep (10 dB input noise)
```
Copies    Input SNR    Output SNR    Gain
     1       +10.0          +8.0    -1.9 dB  (one copy — EAM can't help)
     2       +10.0         +10.2    +0.3 dB  (barely breaks even)
     5       +10.0         +12.3    +2.4 dB  (meaningful denoising)
    10       +10.0         +13.7    +3.7 dB  (solid improvement)
    20       +10.0         +14.4    +4.5 dB  (diminishing returns)
    50       +10.0         +14.9    +4.9 dB  (near saturation)
```

Clear logarithmic scaling — the superposition averages out random noise while the consistent signal reinforces.

### Single-Window Isolation Test
```
Input SNR:         +9.3 dB
SDM (fast):       +20.1 dB  (+10.8 dB improvement)
SDM (iterative):  +19.9 dB  (+10.6 dB improvement)
Direct average:   +23.5 dB  (+14.2 dB — theoretical best)
```

On a single 128-sample segment with 20 noisy copies, EAM achieves **76% of the theoretical optimal** denoising (compared to simply averaging all copies in numpy).

## Critical Design Discoveries

### 1. Position Binding (Holographic Memory)

**Problem:** For periodic signals (sine waves), all windows look similar regardless of position. The EAM can't distinguish "same window, different noise" from "different window, same phase" — it blends everything together, causing phase smearing that's WORSE than the original noise.

**Solution:** Multiply each window vector by a position-dependent random ±1 key before writing. This is "binding" from holographic associative memory — binding content to an address.

```
encode(window, position) = normalize(window) ⊙ position_key[position]
decode(reconstruction, position) = reconstruction ⊙ position_key[position]
```

The ±1 key is self-inverse (multiply twice = identity). Same-position copies stay similar (same key, similar content). Different positions become nearly orthogonal (different random keys).

**Without binding:** -4.1 dB (denoising makes signal WORSE)
**With binding:** +4.9 dB (genuine improvement)

### 2. Encoding Simplicity

**Failed approach:** Feature engineering (64 raw samples + 32 spectral + 32 statistics = 128 dims). The non-linear features (FFT magnitudes, zero crossings, kurtosis) don't average cleanly in superposition. The decode could only recover 64 dims, throwing away half the reconstruction.

**Working approach:** The signal IS the vector. 128 samples → 128 dims, unit normalized. Superposition directly averages waveforms. Encode and decode are trivial (normalize / scale).

**Lesson:** For superposition to work as averaging, the representation must be LINEAR. Feature engineering breaks linearity.

### 3. Amplitude Recovery

**Failed:** Using max(abs) of noisy window to rescale. Noise peaks inflate the amplitude, causing systematic overshoot.

**Working:** Using L2 norm of noisy window. L2 norm is robust to noise (noise power averages out across dimensions).

## Why This Matters

| | Vector DB | SDM (Chorus) |
|---|---|---|
| Store 50 noisy copies | Redundant (returns nearest one) | Each copy reinforces signal |
| Query with noisy input | Returns one noisy copy back | Reconstructs consensus = denoised |
| Benefit of more copies | Zero | SNR improves logarithmically |
| Mechanism | Nearest neighbor lookup | Superposition averaging |

This is the **law of large numbers manifested through associative memory**. Random noise is incoherent across copies and cancels in superposition. The consistent signal is coherent and reinforces. The EAM's weighted-sum reconstruction acts as a statistical filter.

### Biological Analogy

This mirrors how biological neurons average noisy sensory signals. Multiple noisy measurements of the same stimulus are integrated through synaptic superposition, producing a cleaner percept. The EAM hard locations act like neurons that accumulate evidence across exposures.

## Parameters
- Dimension: 128 (all dims = raw signal samples)
- Window: 128 samples, stride 64 (50% overlap)
- Position binding: random ±1 vectors seeded by position index
- Signal: 440Hz sine, 8000 Hz sample rate, 0.5s duration
- SDM: k=20 activation, competitive learning, 1000-2000 adaptive hard locations
