"""Finance experiment engine — Market Déjà Vu + Cross-Asset Completion.

Two experiments:

1. DÉJÀ VU: Store single-asset windows. Walk forward. At each step, query SDM.
   Reconstruction fidelity = how familiar this moment feels.
   The reconstruction itself = what collective memory expects.

2. CROSS-ASSET: Store multi-asset windows. Give partial info (e.g. stocks only).
   SDM completes the pattern for bonds/gold/oil.
   Test: does the completion match reality?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .client import HeatherClient
from .data import MarketData
from .encoding import (
    encode_window, encode_multi_asset, decode_window,
    cosine_similarity,
)


@dataclass
class DejaVuResult:
    """Walk-forward results for Market Déjà Vu."""
    dates: list[str]
    prices: list[float]
    fidelities: list[float]
    predictions: list[list[float]]  # decoded predicted return shapes
    actuals: list[list[float]]      # actual return windows
    prediction_scores: list[float]  # cosine(prediction, actual)

    @property
    def mean_fidelity(self) -> float:
        return float(np.mean(self.fidelities)) if self.fidelities else 0.0

    @property
    def mean_prediction_score(self) -> float:
        return float(np.mean(self.prediction_scores)) if self.prediction_scores else 0.0

    def fidelity_dips(self, threshold: float = None) -> list[tuple[str, float]]:
        """Find dates where fidelity dropped significantly.
        Default threshold: 1 std below mean."""
        if not self.fidelities:
            return []
        if threshold is None:
            mu = np.mean(self.fidelities)
            sigma = np.std(self.fidelities)
            threshold = mu - sigma
        return [(d, f) for d, f in zip(self.dates, self.fidelities)
                if f < threshold]


@dataclass
class CrossAssetResult:
    """Results for cross-asset pattern completion."""
    date: str
    known_assets: dict[str, list[float]]   # assets provided as input
    predicted_assets: dict[str, list[float]]  # EAM's completion
    actual_assets: dict[str, list[float]]   # ground truth
    similarities: dict[str, float]           # per-asset prediction quality
    fidelity: float


class Finance:
    """SDM-powered financial analysis engine."""

    def __init__(self, heather: HeatherClient, dimension: int = 128):
        self.heather = heather
        self.dimension = dimension
        self.windows_stored = 0

    # ── Market Déjà Vu ──────────────────────────────────────────

    def ingest_single(self, data: MarketData, ticker: str,
                      window: int = 20, stride: int = 1) -> int:
        """Store single-asset windows in SDM."""
        rets = data.returns(ticker)
        batch = []

        for i in range(0, len(rets) - window + 1, stride):
            w = rets[i:i + window]
            vec = encode_window(w, target_dims=self.dimension)
            batch.append(vec)

            if len(batch) >= 100:
                self.heather.write(batch)
                self.windows_stored += len(batch)
                batch = []

        if batch:
            self.heather.write(batch)
            self.windows_stored += len(batch)

        return self.windows_stored

    def dejavu(self, data: MarketData, ticker: str,
               train_pct: float = 0.7, window: int = 20) -> DejaVuResult:
        """Run Market Déjà Vu experiment.

        Train on first train_pct of data, walk forward on the rest.
        At each step: query SDM with current window, measure fidelity,
        decode reconstruction as "expected next."
        """
        rets = data.returns(ticker)
        prices = data.assets[ticker]
        n = len(rets)
        train_end = int(n * train_pct)

        # Ingest training portion
        batch = []
        for i in range(0, train_end - window + 1):
            w = rets[i:i + window]
            vec = encode_window(w, target_dims=self.dimension)
            batch.append(vec)
            if len(batch) >= 100:
                self.heather.write(batch)
                self.windows_stored += len(batch)
                batch = []
        if batch:
            self.heather.write(batch)
            self.windows_stored += len(batch)

        # Walk forward
        dates = []
        fidelities = []
        predictions = []
        actuals = []
        pred_scores = []
        test_prices = []

        for i in range(train_end, n - window):
            current_window = rets[i:i + window]
            next_window = rets[i + 1:i + 1 + window] if i + 1 + window <= n else None

            vec = encode_window(current_window, target_dims=self.dimension)
            reconstructed = self.heather.read(vec)

            fidelity = cosine_similarity(vec, reconstructed)
            predicted_shape = decode_window(reconstructed)

            dates.append(data.dates[i + window])  # date of last day in window
            test_prices.append(prices[i + window])
            fidelities.append(fidelity)
            predictions.append(predicted_shape)
            actuals.append(current_window)

            # Prediction quality: compare reconstruction with next window
            if next_window and len(next_window) == window:
                next_vec = encode_window(next_window, target_dims=self.dimension)
                pred_score = cosine_similarity(reconstructed, next_vec)
                pred_scores.append(pred_score)
            else:
                pred_scores.append(0.0)

        return DejaVuResult(
            dates=dates,
            prices=test_prices,
            fidelities=fidelities,
            predictions=predictions,
            actuals=actuals,
            prediction_scores=pred_scores,
        )

    # ── Cross-Asset Completion ──────────────────────────────────

    def ingest_multi(self, data: MarketData, window: int = 20,
                     stride: int = 1) -> int:
        """Store multi-asset windows in SDM."""
        tickers = sorted(data.tickers)
        all_returns = {t: data.returns(t) for t in tickers}
        min_len = min(len(r) for r in all_returns.values())

        batch = []
        for i in range(0, min_len - window + 1, stride):
            asset_rets = {t: all_returns[t][i:i + window] for t in tickers}
            vec = encode_multi_asset(asset_rets, target_dims=self.dimension)
            batch.append(vec)

            if len(batch) >= 100:
                self.heather.write(batch)
                self.windows_stored += len(batch)
                batch = []

        if batch:
            self.heather.write(batch)
            self.windows_stored += len(batch)

        return self.windows_stored

    def complete_assets(self, data: MarketData, known_ticker: str,
                        window_idx: int, window: int = 20) -> CrossAssetResult:
        """Given one asset's returns, predict what the others are doing.

        Encode the known asset in its partition, zero the others,
        query SDM. The reconstruction completes the missing assets.
        """
        tickers = sorted(data.tickers)
        all_returns = {t: data.returns(t) for t in tickers}
        n_assets = len(tickers)
        dims_per_asset = self.dimension // n_assets

        # Build partial vector (known asset filled, others zeroed)
        known_idx = tickers.index(known_ticker)
        known_rets = all_returns[known_ticker][window_idx:window_idx + window]

        partial = np.zeros(self.dimension, dtype=np.float64)
        known_encoded = encode_window(known_rets, target_dims=dims_per_asset)
        start = known_idx * dims_per_asset
        partial[start:start + dims_per_asset] = known_encoded

        norm = np.linalg.norm(partial)
        if norm > 0:
            partial = partial / norm

        # Query SDM
        reconstructed = self.heather.read(partial.tolist())
        fidelity = cosine_similarity(partial.tolist(), reconstructed)

        # Decode each asset from reconstruction
        rec = np.array(reconstructed, dtype=np.float64)
        predicted_assets = {}
        actual_assets = {}
        similarities = {}

        for i, ticker in enumerate(tickers):
            asset_start = i * dims_per_asset
            asset_vec = rec[asset_start:asset_start + dims_per_asset].tolist()
            predicted_shape = decode_window(asset_vec, n_returns=min(window, dims_per_asset))
            predicted_assets[ticker] = predicted_shape

            actual_rets = all_returns[ticker][window_idx:window_idx + window]
            actual_assets[ticker] = actual_rets

            # Compare predicted vs actual
            if len(actual_rets) == window:
                actual_vec = encode_window(actual_rets, target_dims=dims_per_asset)
                sim = cosine_similarity(asset_vec, actual_vec)
                similarities[ticker] = sim

        known_assets = {known_ticker: known_rets}
        date = data.dates[window_idx + window] if window_idx + window < len(data.dates) else "?"

        return CrossAssetResult(
            date=date,
            known_assets=known_assets,
            predicted_assets=predicted_assets,
            actual_assets=actual_assets,
            similarities=similarities,
            fidelity=fidelity,
        )
