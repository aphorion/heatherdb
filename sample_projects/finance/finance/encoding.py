"""Encode time-series windows as high-dimensional vectors for SDM.

Single-asset encoding: 20-day return window → 128 dims
Multi-asset encoding: 4 assets × 32 dims each = 128 dims
"""

from __future__ import annotations

import numpy as np


def encode_window(returns: list[float], target_dims: int = 128) -> list[float]:
    """Encode a single-asset return window into a vector.

    Features:
    - Raw normalized returns (20 dims)
    - Cumulative return curve (20 dims)
    - Absolute returns / volatility shape (20 dims)
    - Rolling 5-day stats (20 dims)
    - Summary statistics (28 dims)
    Total: 128 dims
    """
    r = np.array(returns, dtype=np.float64)
    n = len(r)

    # Normalize returns (z-score within window)
    mu, sigma = np.mean(r), np.std(r)
    if sigma > 0:
        r_norm = (r - mu) / sigma
    else:
        r_norm = r - mu

    features = []

    # 1. Raw normalized returns (pad/truncate to 20)
    padded = np.zeros(20)
    padded[:min(n, 20)] = r_norm[:20]
    features.extend(padded.tolist())

    # 2. Cumulative return curve
    cum = np.cumsum(r)
    cum_padded = np.zeros(20)
    cum_padded[:min(n, 20)] = cum[:20]
    # Normalize cumulative
    cum_max = np.max(np.abs(cum_padded))
    if cum_max > 0:
        cum_padded = cum_padded / cum_max
    features.extend(cum_padded.tolist())

    # 3. Absolute returns (volatility shape)
    abs_r = np.abs(r)
    abs_padded = np.zeros(20)
    abs_padded[:min(n, 20)] = abs_r[:20]
    abs_max = np.max(abs_padded)
    if abs_max > 0:
        abs_padded = abs_padded / abs_max
    features.extend(abs_padded.tolist())

    # 4. Rolling 5-day statistics (mean + std, 10 points each = 20 dims)
    roll_mean = []
    roll_std = []
    for i in range(0, min(n, 20) - 4):
        w = r[i:i + 5]
        roll_mean.append(np.mean(w))
        roll_std.append(np.std(w))
    # Pad to 10 each
    rm = np.zeros(10)
    rs = np.zeros(10)
    rm[:min(len(roll_mean), 10)] = roll_mean[:10]
    rs[:min(len(roll_std), 10)] = roll_std[:10]
    # Normalize
    rm_max = np.max(np.abs(rm))
    if rm_max > 0:
        rm = rm / rm_max
    rs_max = np.max(rs)
    if rs_max > 0:
        rs = rs / rs_max
    features.extend(rm.tolist())
    features.extend(rs.tolist())

    # 5. Summary statistics (28 dims)
    total_return = np.sum(r)
    volatility = np.std(r)
    skewness = float(np.mean(r_norm ** 3)) if sigma > 0 else 0.0
    kurtosis = float(np.mean(r_norm ** 4) - 3.0) if sigma > 0 else 0.0
    max_drawdown = _max_drawdown(r)
    up_ratio = np.mean(r > 0)
    max_ret = np.max(r) if n > 0 else 0.0
    min_ret = np.min(r) if n > 0 else 0.0
    ret_range = max_ret - min_ret

    # Trend: linear regression slope
    if n > 1:
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, r, 1)[0]
    else:
        slope = 0.0

    # Autocorrelation lag-1
    if n > 2 and sigma > 0:
        autocorr = np.corrcoef(r[:-1], r[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    # Mean reversion indicator
    half_n = n // 2
    if half_n > 0:
        first_half = np.mean(r[:half_n])
        second_half = np.mean(r[half_n:])
        mean_reversion = second_half - first_half
    else:
        mean_reversion = 0.0

    summary = [
        total_return, volatility, skewness, kurtosis,
        max_drawdown, up_ratio, max_ret, min_ret, ret_range,
        slope, autocorr, mean_reversion,
        # Repeat key features scaled differently to fill 28 dims
        np.tanh(total_return * 10),
        np.tanh(volatility * 50),
        np.tanh(skewness),
        np.tanh(kurtosis / 3),
        1.0 if total_return > 0 else -1.0,
        1.0 if slope > 0 else -1.0,
        abs(total_return),
        abs(max_drawdown),
        min(volatility * 20, 1.0),
        max(0, up_ratio - 0.5) * 2,
        np.sign(autocorr) * min(abs(autocorr), 1.0),
        np.tanh(mean_reversion * 20),
        # Interaction terms
        total_return * volatility,
        slope * volatility,
        skewness * kurtosis,
        up_ratio * total_return,
    ]
    features.extend(summary)

    # Ensure exactly target_dims
    vec = np.array(features[:target_dims], dtype=np.float64)
    if len(vec) < target_dims:
        vec = np.pad(vec, (0, target_dims - len(vec)))

    # Normalize to unit vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


def encode_multi_asset(asset_returns: dict[str, list[float]],
                       target_dims: int = 128) -> list[float]:
    """Encode multiple assets' returns in one window into a single vector.
    Each asset gets target_dims / num_assets dimensions.

    This creates cross-asset patterns: when stocks crash, what do bonds do?
    The EAM can complete the pattern across assets.
    """
    tickers = sorted(asset_returns.keys())
    n_assets = len(tickers)
    dims_per_asset = target_dims // n_assets

    features = []
    for ticker in tickers:
        rets = asset_returns[ticker]
        asset_vec = encode_window(rets, target_dims=dims_per_asset)
        features.extend(asset_vec)

    # Pad if needed
    vec = np.array(features[:target_dims], dtype=np.float64)
    if len(vec) < target_dims:
        vec = np.pad(vec, (0, target_dims - len(vec)))

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


def decode_window(vec: list[float], n_returns: int = 20) -> list[float]:
    """Decode a vector back to approximate returns (first 20 dims).
    This is lossy — we only recover the normalized return shape."""
    v = np.array(vec[:n_returns], dtype=np.float64)
    # Denormalize from unit vector
    max_val = np.max(np.abs(v))
    if max_val > 0:
        v = v / max_val
    return v.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd)) if len(dd) > 0 else 0.0
