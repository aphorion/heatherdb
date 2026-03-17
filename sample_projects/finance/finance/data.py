"""Market data loading — real (yfinance) and synthetic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass
class MarketData:
    """Price series for one or more assets."""
    dates: list[str]           # ISO date strings
    assets: dict[str, list[float]]  # ticker → daily prices
    source: str                # "yfinance" or "synthetic"

    def returns(self, ticker: str) -> list[float]:
        """Daily log returns."""
        prices = self.assets[ticker]
        return [np.log(prices[i] / prices[i - 1])
                for i in range(1, len(prices))]

    @property
    def tickers(self) -> list[str]:
        return list(self.assets.keys())

    @property
    def length(self) -> int:
        return len(self.dates)


def load_yfinance(tickers: list[str], years: int = 5) -> MarketData:
    """Load real market data via yfinance."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), auto_adjust=True)

    if len(tickers) == 1:
        prices = data["Close"].dropna()
        dates = [d.strftime("%Y-%m-%d") for d in prices.index]
        assets = {tickers[0]: prices.values.tolist()}
    else:
        prices = data["Close"].dropna()
        dates = [d.strftime("%Y-%m-%d") for d in prices.index]
        assets = {}
        for t in tickers:
            assets[t] = prices[t].values.tolist()

    return MarketData(dates=dates, assets=assets, source="yfinance")


def generate_synthetic(days: int = 2000, seed: int = 42) -> MarketData:
    """Generate synthetic multi-asset data with known regime changes and crashes.

    Assets: STOCK, BOND, GOLD, OIL — with realistic correlations.
    Includes 3 crash events and 2 regime shifts for testing.
    """
    rng = np.random.RandomState(seed)

    # Base parameters
    dt = 1 / 252  # daily

    # Regime periods (index ranges)
    # 0-500: bull (low vol, positive drift)
    # 500-700: crash 1 (sharp drop in stocks, flight to bonds/gold)
    # 700-1200: recovery
    # 1200-1400: crash 2
    # 1400-1700: high-vol sideways
    # 1700-1850: crash 3
    # 1850-2000: final recovery

    stock_prices = [100.0]
    bond_prices = [100.0]
    gold_prices = [100.0]
    oil_prices = [50.0]

    for i in range(1, days):
        # Determine regime
        if i < 500:
            # Bull market
            s_drift, s_vol = 0.15, 0.12
            b_drift, b_vol = 0.03, 0.04
            g_drift, g_vol = 0.02, 0.08
            o_drift, o_vol = 0.05, 0.15
            corr_sb = -0.2  # stocks-bonds
        elif i < 600:
            # Crash 1 — stocks tank, bonds/gold rally
            s_drift, s_vol = -0.8, 0.35
            b_drift, b_vol = 0.15, 0.06
            g_drift, g_vol = 0.25, 0.12
            o_drift, o_vol = -0.5, 0.30
            corr_sb = -0.6
        elif i < 700:
            # Crash 1 tail — volatility fading
            s_drift, s_vol = -0.2, 0.25
            b_drift, b_vol = 0.08, 0.05
            g_drift, g_vol = 0.10, 0.10
            o_drift, o_vol = -0.15, 0.22
            corr_sb = -0.4
        elif i < 1200:
            # Recovery
            s_drift, s_vol = 0.20, 0.15
            b_drift, b_vol = 0.02, 0.04
            g_drift, g_vol = 0.01, 0.08
            o_drift, o_vol = 0.10, 0.18
            corr_sb = -0.15
        elif i < 1350:
            # Crash 2
            s_drift, s_vol = -0.6, 0.40
            b_drift, b_vol = 0.12, 0.07
            g_drift, g_vol = 0.30, 0.15
            o_drift, o_vol = -0.7, 0.35
            corr_sb = -0.7
        elif i < 1700:
            # High-vol sideways
            s_drift, s_vol = 0.02, 0.22
            b_drift, b_vol = 0.04, 0.05
            g_drift, g_vol = 0.05, 0.10
            o_drift, o_vol = 0.0, 0.20
            corr_sb = -0.3
        elif i < 1850:
            # Crash 3
            s_drift, s_vol = -0.5, 0.35
            b_drift, b_vol = 0.10, 0.06
            g_drift, g_vol = 0.20, 0.12
            o_drift, o_vol = -0.4, 0.28
            corr_sb = -0.6
        else:
            # Final recovery
            s_drift, s_vol = 0.18, 0.14
            b_drift, b_vol = 0.03, 0.04
            g_drift, g_vol = 0.02, 0.08
            o_drift, o_vol = 0.08, 0.16
            corr_sb = -0.2

        # Generate correlated returns
        z = rng.randn(4)
        # Apply simple correlation structure
        z_stock = z[0]
        z_bond = corr_sb * z[0] + np.sqrt(1 - corr_sb ** 2) * z[1]
        z_gold = -0.1 * z[0] + 0.3 * z[1] + np.sqrt(1 - 0.1**2 - 0.3**2) * z[2]
        z_oil = 0.3 * z[0] + 0.1 * z[1] + np.sqrt(1 - 0.3**2 - 0.1**2) * z[3]

        s_ret = s_drift * dt + s_vol * np.sqrt(dt) * z_stock
        b_ret = b_drift * dt + b_vol * np.sqrt(dt) * z_bond
        g_ret = g_drift * dt + g_vol * np.sqrt(dt) * z_gold
        o_ret = o_drift * dt + o_vol * np.sqrt(dt) * z_oil

        stock_prices.append(stock_prices[-1] * np.exp(s_ret))
        bond_prices.append(bond_prices[-1] * np.exp(b_ret))
        gold_prices.append(gold_prices[-1] * np.exp(g_ret))
        oil_prices.append(oil_prices[-1] * np.exp(o_ret))

    # Generate dates
    base = datetime(2015, 1, 2)
    dates = [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    return MarketData(
        dates=dates,
        assets={
            "STOCK": stock_prices,
            "BOND": bond_prices,
            "GOLD": gold_prices,
            "OIL": oil_prices,
        },
        source="synthetic",
    )


# Known crash windows in synthetic data (for validation)
SYNTHETIC_CRASHES = [
    (500, 700, "Crash 1"),
    (1200, 1350, "Crash 2"),
    (1700, 1850, "Crash 3"),
]
