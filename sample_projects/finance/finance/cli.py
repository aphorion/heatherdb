"""Interactive terminal for Finance — Market Déjà Vu + Cross-Asset Completion."""

from __future__ import annotations

import random

from .engine import Finance, DejaVuResult
from .data import MarketData, load_yfinance, generate_synthetic

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def spark_line(values: list[float], width: int = 60) -> str:
    """ASCII sparkline."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    line = ""
    step = max(1, len(values) // width)
    for i in range(0, len(values), step):
        v = values[i]
        idx = int((v - mn) / rng * (len(blocks) - 1))
        line += blocks[idx]
    return line


def fidelity_color(f: float) -> str:
    if f > 0.8:
        return GREEN
    if f > 0.6:
        return YELLOW
    return RED


def cmd_load(args: str) -> MarketData | None:
    """Load market data from yfinance."""
    parts = args.split()
    if not parts:
        print(f"{DIM}  usage: load <ticker1,ticker2,...> [years]{RESET}")
        print(f"{DIM}  example: load SPY,TLT,GLD,USO 10{RESET}\n")
        return None

    tickers = [t.strip().upper() for t in parts[0].split(",")]
    years = int(parts[1]) if len(parts) > 1 else 5

    print(f"\n{BOLD}Loading {', '.join(tickers)} ({years} years)...{RESET}")
    try:
        data = load_yfinance(tickers, years=years)
        print(f"{GREEN}  Loaded {data.length} days for {len(data.tickers)} assets{RESET}")
        for t in data.tickers:
            prices = data.assets[t]
            print(f"  {DIM}{t}: ${prices[0]:.2f} → ${prices[-1]:.2f}  "
                  f"({(prices[-1]/prices[0]-1)*100:+.1f}%){RESET}")
        print()
        return data
    except Exception as e:
        print(f"{RED}  error loading data: {e}{RESET}\n")
        return None


def cmd_dejavu(finance: Finance, data: MarketData, ticker: str,
               train_pct: float = 0.7):
    """Run Market Déjà Vu experiment."""
    if ticker not in data.tickers:
        print(f"{RED}  ticker '{ticker}' not in loaded data. "
              f"Available: {', '.join(data.tickers)}{RESET}\n")
        return None

    prices = data.assets[ticker]
    n = len(prices)
    train_end = int(n * train_pct)

    print(f"\n{BOLD}Market Déjà Vu — {ticker}{RESET}")
    print(f"{DIM}Training on first {train_pct:.0%} ({train_end} days), "
          f"testing on remaining {n - train_end} days{RESET}")
    print(f"{DIM}Storing windows...{RESET}")

    result = finance.dejavu(data, ticker, train_pct=train_pct)

    print(f"{GREEN}  {finance.windows_stored} windows stored in SDM{RESET}\n")

    # Fidelity timeline
    print(f"{BOLD}Fidelity Timeline{RESET} — how familiar each moment feels")
    print(f"{DIM}(low = SDM doesn't recognize this pattern = unusual market){RESET}\n")

    # ASCII chart: price on top, fidelity below
    _print_dual_chart(result)

    # Summary stats
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  {DIM}Mean fidelity: {result.mean_fidelity:.4f}{RESET}")
    print(f"  {DIM}Mean prediction score: {result.mean_prediction_score:.4f}{RESET}")

    # Fidelity dips
    dips = result.fidelity_dips()
    if dips:
        print(f"\n{BOLD}Significant fidelity dips{RESET} (unusual market moments):")
        for date, fid in dips[:10]:
            color = fidelity_color(fid)
            print(f"  {color}  {date}  fidelity: {fid:.4f}{RESET}")

    print()
    return result


def _print_dual_chart(result: DejaVuResult, width: int = 70):
    """Print price chart and fidelity chart aligned."""
    if not result.dates:
        return

    # Price sparkline
    print(f"  {DIM}Price:{RESET}    {spark_line(result.prices, width)}")
    print(f"  {DIM}         {result.dates[0]:>10s}"
          f"{'':>{width - 22}}{result.dates[-1]:<10s}{RESET}")

    # Fidelity sparkline with color
    fids = result.fidelities
    mn, mx = min(fids), max(fids)
    rng = mx - mn if mx > mn else 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    step = max(1, len(fids) // width)

    fid_line = ""
    for i in range(0, len(fids), step):
        f = fids[i]
        idx = int((f - mn) / rng * (len(blocks) - 1))
        color = fidelity_color(f)
        fid_line += f"{color}{blocks[idx]}{RESET}"

    print(f"  {DIM}Fidelity:{RESET} {fid_line}")
    print(f"  {DIM}         low={mn:.3f}  high={mx:.3f}  "
          f"mean={result.mean_fidelity:.3f}{RESET}")


def cmd_crossasset(finance: Finance, data: MarketData,
                   known_ticker: str):
    """Run Cross-Asset Completion experiment."""
    if known_ticker not in data.tickers:
        print(f"{RED}  '{known_ticker}' not in data. "
              f"Available: {', '.join(data.tickers)}{RESET}\n")
        return

    if len(data.tickers) < 2:
        print(f"{RED}  need at least 2 assets for cross-asset completion{RESET}\n")
        return

    # First ingest all multi-asset windows
    print(f"\n{BOLD}Cross-Asset Completion{RESET}")
    print(f"{DIM}Given only {known_ticker}, SDM predicts what other assets are doing{RESET}")
    print(f"{DIM}Ingesting multi-asset windows...{RESET}")

    finance.ingest_multi(data, window=20, stride=5)
    print(f"{GREEN}  {finance.windows_stored} multi-asset windows stored{RESET}\n")

    # Test at several random points in the second half
    rets = data.returns(known_ticker)
    n = len(rets)
    test_points = sorted(random.sample(range(n // 2, n - 25), min(5, (n - 25 - n // 2))))

    for idx in test_points:
        result = finance.complete_assets(data, known_ticker, idx)

        print(f"  {BOLD}Date: {result.date}{RESET}  "
              f"{DIM}fidelity: {result.fidelity:.3f}{RESET}")

        for ticker in sorted(result.similarities.keys()):
            sim = result.similarities[ticker]
            color = GREEN if sim > 0.5 else YELLOW if sim > 0.2 else RED
            is_known = " (input)" if ticker == known_ticker else " (predicted)"
            actual_rets = result.actual_assets.get(ticker, [])
            actual_dir = "↑" if sum(actual_rets) > 0 else "↓" if actual_rets else "?"
            pred_rets = result.predicted_assets.get(ticker, [])
            pred_dir = "↑" if sum(pred_rets) > 0 else "↓" if pred_rets else "?"
            match = "✓" if actual_dir == pred_dir else "✗"

            print(f"    {color}{ticker:6s}{RESET} sim={sim:+.3f}  "
                  f"actual:{actual_dir} predicted:{pred_dir} {match}{is_known}")

    print()


def cmd_predict(finance: Finance, data: MarketData, ticker: str):
    """Show what SDM expects vs what actually happened for recent windows."""
    if ticker not in data.tickers:
        print(f"{RED}  '{ticker}' not in data{RESET}\n")
        return

    rets = data.returns(ticker)
    n = len(rets)

    print(f"\n{BOLD}SDM Predictions vs Reality — {ticker}{RESET}")
    print(f"{DIM}What collective memory expected vs what happened{RESET}\n")

    # Test last 5 windows
    from .encoding import encode_window, decode_window, cosine_similarity
    for offset in range(5):
        idx = n - 25 + offset * 5
        if idx < 20 or idx + 20 > n:
            continue

        window = rets[idx:idx + 20]
        vec = encode_window(window, target_dims=finance.dimension)
        reconstructed = finance.heather.read(vec)

        fidelity = cosine_similarity(vec, reconstructed)
        predicted = decode_window(reconstructed)
        actual = window

        date = data.dates[idx + 20] if idx + 20 < len(data.dates) else "?"

        # Sparklines
        actual_spark = spark_line(actual, 30)
        pred_spark = spark_line(predicted[:len(actual)], 30)

        color = fidelity_color(fidelity)
        print(f"  {BOLD}{date}{RESET}  {DIM}fidelity: {color}{fidelity:.3f}{RESET}")
        print(f"    actual:    {actual_spark}  "
              f"{DIM}ret={sum(actual)*100:+.1f}%{RESET}")
        print(f"    predicted: {pred_spark}  "
              f"{DIM}shape match{RESET}")
        print()


def run(finance: Finance):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Finance{RESET} — Market Déjà Vu + Cross-Asset Completion")
    print(f"{DIM}SDM stores market patterns. Query with now → it predicts what comes next.{RESET}")
    print(f"{DIM}Commands:{RESET}")
    print(f"{DIM}  load <tickers> [years]   Load data (e.g. load SPY,TLT,GLD,USO 10){RESET}")
    print(f"{DIM}  synthetic                Use synthetic multi-asset data{RESET}")
    print(f"{DIM}  dejavu <ticker>          Market Déjà Vu — fidelity walk-forward{RESET}")
    print(f"{DIM}  cross <ticker>           Cross-asset completion (give 1, predict rest){RESET}")
    print(f"{DIM}  predict <ticker>         Show SDM predictions vs reality{RESET}")
    print(f"{DIM}  stats                    SDM statistics{RESET}")
    print(f"{DIM}  quit                     Exit{RESET}\n")

    data: MarketData | None = None

    while True:
        try:
            user_input = input(f"{CYAN}fin>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        parts = user_input.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if cmd == "load":
            result = cmd_load(args)
            if result:
                data = result
            continue

        if cmd == "synthetic":
            print(f"\n{BOLD}Generating synthetic data...{RESET}")
            data = generate_synthetic()
            print(f"{GREEN}  Generated {data.length} days for "
                  f"{', '.join(data.tickers)}{RESET}")
            for t in data.tickers:
                prices = data.assets[t]
                print(f"  {DIM}{t}: ${prices[0]:.2f} → ${prices[-1]:.2f}{RESET}")
            print()
            continue

        if cmd == "dejavu":
            if not data:
                print(f"{DIM}  load data first (load or synthetic){RESET}\n")
                continue
            ticker = args.strip().upper() or data.tickers[0]
            try:
                cmd_dejavu(finance, data, ticker)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "cross":
            if not data:
                print(f"{DIM}  load data first{RESET}\n")
                continue
            ticker = args.strip().upper() or data.tickers[0]
            try:
                cmd_crossasset(finance, data, ticker)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "predict":
            if not data:
                print(f"{DIM}  load data first{RESET}\n")
                continue
            ticker = args.strip().upper() or data.tickers[0]
            try:
                cmd_predict(finance, data, ticker)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "stats":
            try:
                stats = finance.heather.stats()
                print(f"\n{DIM}  Windows stored: {finance.windows_stored}")
                if data:
                    print(f"  Data: {data.source}, {data.length} days, "
                          f"{', '.join(data.tickers)}")
                print(f"  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        print(f"{DIM}  unknown command. try: load, synthetic, dejavu, cross, predict, stats, quit{RESET}\n")
