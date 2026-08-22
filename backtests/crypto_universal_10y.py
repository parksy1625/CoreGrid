import csv
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

START = pd.Timestamp("2016-08-22 00:00:00", tz="UTC")
END = pd.Timestamp("2026-08-22 00:00:00", tz="UTC")
INITIAL_CASH = 10_000_000.0
FEE = 0.0005  # 0.05% each side
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "AVAXUSDT",
]
API_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
]
HOUR_MS = 3_600_000

OUT = Path("backtest_output")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("backtest_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def ts_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def request_klines(session: requests.Session, symbol: str, start_ms: int, end_ms: int):
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    errors = []
    for endpoint in API_ENDPOINTS:
        for attempt in range(4):
            try:
                r = session.get(endpoint, params=params, timeout=35)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        return data, endpoint
                    errors.append(f"{endpoint}: unexpected JSON")
                    break
                if r.status_code in (418, 429):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                errors.append(f"{endpoint}: HTTP {r.status_code} {r.text[:120]}")
                break
            except Exception as e:
                errors.append(f"{endpoint}: {type(e).__name__}: {e}")
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(" | ".join(errors[-8:]))


def fetch_symbol(symbol: str) -> tuple[pd.DataFrame, str]:
    cache_path = CACHE / f"{symbol}_1h.csv"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        return df, "cache"

    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-universal-backtest/1.0"})
    start_ms = ts_ms(START)
    end_ms = ts_ms(END) - 1
    rows = []
    endpoint_used = None
    no_progress = 0

    while start_ms <= end_ms:
        data, ep = request_klines(session, symbol, start_ms, end_ms)
        endpoint_used = ep
        if not data:
            # If a query window predates listing, move forward by the maximum
            # response span (~1000 hours) rather than aborting.
            start_ms += 1000 * HOUR_MS
            no_progress += 1
            if no_progress > 200:
                break
            continue
        no_progress = 0
        for r in data:
            # [openTime, open, high, low, close, volume, closeTime, ...]
            rows.append((
                int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                float(r[4]), float(r[5]),
            ))
        last_open = int(data[-1][0])
        next_start = last_open + HOUR_MS
        if next_start <= start_ms:
            raise RuntimeError(f"{symbol}: API made no progress at {start_ms}")
        start_ms = next_start
        time.sleep(0.035)

    if not rows:
        raise RuntimeError(f"{symbol}: no 1h data returned")

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[(df["datetime"] >= START) & (df["datetime"] < END)]
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df.to_csv(cache_path, index_label="datetime")
    return df, endpoint_used or "unknown"


def to_4h(df: pd.DataFrame) -> pd.DataFrame:
    out = df.resample("4h", origin="epoch", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return out.dropna(subset=["open", "high", "low", "close"])


def wilder_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    if n > 1:
        tr[1:] = np.maximum.reduce([
            h[1:] - l[1:],
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1]),
        ])
    atr = np.full(n, np.nan, dtype=float)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def supertrend_target(df: pd.DataFrame, period: int, mult: float) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = wilder_atr(df, period)
    n = len(df)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    st = np.full(n, np.nan)
    bull = np.zeros(n, dtype=bool)

    first = period - 1
    if n <= first:
        return bull
    for i in range(first, n):
        if not np.isfinite(atr[i]):
            continue
        hl2 = (h[i] + l[i]) / 2.0
        bu = hl2 + mult * atr[i]
        bl = hl2 - mult * atr[i]
        if i == first or not np.isfinite(fu[i - 1]):
            fu[i] = bu
            fl[i] = bl
            # Conventional initialization: price at/above midpoint starts bullish.
            st[i] = fl[i] if c[i] >= hl2 else fu[i]
        else:
            prev_fu, prev_fl, prev_st = fu[i - 1], fl[i - 1], st[i - 1]
            prev_close = c[i - 1]
            fu[i] = bu if (bu < prev_fu or prev_close > prev_fu) else prev_fu
            fl[i] = bl if (bl > prev_fl or prev_close < prev_fl) else prev_fl
            if np.isclose(prev_st, prev_fu, rtol=1e-12, atol=1e-12):
                st[i] = fu[i] if c[i] <= fu[i] else fl[i]
            else:
                st[i] = fl[i] if c[i] >= fl[i] else fu[i]
        bull[i] = np.isfinite(st[i]) and np.isclose(st[i], fl[i], rtol=1e-10, atol=1e-10)
    return bull


def donchian_target(df: pd.DataFrame, entry_n: int, exit_n: int) -> np.ndarray:
    close = df["close"].to_numpy(float)
    prev_high = df["high"].shift(1).rolling(entry_n).max().to_numpy(float)
    prev_low = df["low"].shift(1).rolling(exit_n).min().to_numpy(float)
    target = np.zeros(len(df), dtype=bool)
    pos = False
    for i in range(len(df)):
        if not pos and np.isfinite(prev_high[i]) and close[i] > prev_high[i]:
            pos = True
        elif pos and np.isfinite(prev_low[i]) and close[i] < prev_low[i]:
            pos = False
        target[i] = pos
    return target


def backtest(df: pd.DataFrame, target: np.ndarray, leverage: float, timeframe_hours: int) -> dict:
    n = len(df)
    if n < 5:
        raise ValueError("not enough bars")
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    realized = INITIAL_CASH
    in_pos = False
    units = 0.0
    entry_px = 0.0
    base_equity = realized
    trade_start_equity = realized
    trade_entry_time = None
    trades = []
    liquidated = False
    eq = np.full(n, INITIAL_CASH, dtype=float)

    for i in range(1, n):
        desired = bool(target[i - 1])

        # Execute previous-close signal at current open.
        if in_pos and not desired:
            pre_exit = base_equity + units * (o[i] - entry_px)
            exit_fee = abs(units * o[i]) * FEE
            realized = max(0.0, pre_exit - exit_fee)
            trades.append({
                "entry_time": str(trade_entry_time),
                "exit_time": str(df.index[i]),
                "entry_price": entry_px,
                "exit_price": o[i],
                "net_pnl": realized - trade_start_equity,
                "return_pct": (realized / trade_start_equity - 1.0) * 100.0 if trade_start_equity > 0 else -100.0,
                "liquidated": False,
            })
            in_pos = False
            units = 0.0
            if realized <= 0:
                eq[i:] = 0.0
                liquidated = True
                break

        if (not in_pos) and desired and realized > 0:
            trade_start_equity = realized
            notional = realized * leverage
            entry_fee = notional * FEE
            base_equity = realized - entry_fee
            units = notional / o[i]
            entry_px = o[i]
            trade_entry_time = df.index[i]
            in_pos = True

        if in_pos:
            # Simplified full-account liquidation check for leveraged long exposure.
            equity_at_low = base_equity + units * (l[i] - entry_px)
            if equity_at_low <= 0:
                realized = 0.0
                trades.append({
                    "entry_time": str(trade_entry_time),
                    "exit_time": str(df.index[i]),
                    "entry_price": entry_px,
                    "exit_price": float(l[i]),
                    "net_pnl": -trade_start_equity,
                    "return_pct": -100.0,
                    "liquidated": True,
                })
                eq[i:] = 0.0
                in_pos = False
                liquidated = True
                break
            eq[i] = base_equity + units * (c[i] - entry_px)
        else:
            eq[i] = realized

    if in_pos and not liquidated:
        final_px = c[-1]
        pre_exit = base_equity + units * (final_px - entry_px)
        exit_fee = abs(units * final_px) * FEE
        realized = max(0.0, pre_exit - exit_fee)
        trades.append({
            "entry_time": str(trade_entry_time),
            "exit_time": str(df.index[-1]),
            "entry_price": entry_px,
            "exit_price": final_px,
            "net_pnl": realized - trade_start_equity,
            "return_pct": (realized / trade_start_equity - 1.0) * 100.0 if trade_start_equity > 0 else -100.0,
            "liquidated": False,
        })
        eq[-1] = realized

    final_equity = float(eq[-1])
    total_return = (final_equity / INITIAL_CASH - 1.0) * 100.0
    years = max((df.index[-1] - df.index[0]).total_seconds() / (365.2425 * 86400.0), 1e-9)
    cagr = ((final_equity / INITIAL_CASH) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    peak = np.maximum.accumulate(eq)
    dd = np.divide(eq, peak, out=np.ones_like(eq), where=peak != 0) - 1.0
    mdd = float(np.nanmin(dd) * 100.0)

    rets = np.zeros(n - 1, dtype=float)
    prev = eq[:-1]
    cur = eq[1:]
    ok = prev > 0
    rets[ok] = cur[ok] / prev[ok] - 1.0
    rets = rets[np.isfinite(rets)]
    if len(rets) > 1 and np.std(rets, ddof=1) > 0:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(365.2425 * 24.0 / timeframe_hours))
    else:
        sharpe = 0.0

    pnls = np.array([t["net_pnl"] for t in trades], dtype=float)
    gross_profit = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if len(pnls) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    wins = int((pnls > 0).sum()) if len(pnls) else 0
    win_rate = (wins / len(pnls) * 100.0) if len(pnls) else 0.0

    return {
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "mdd_pct": mdd,
        "profit_factor": pf,
        "sharpe": sharpe,
        "win_rate_pct": win_rate,
        "trades": int(len(trades)),
        "liquidations": int(sum(bool(t["liquidated"]) for t in trades)),
        "final_equity_krw": final_equity,
    }


def buy_hold(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {}
    r = df["close"].iloc[-1] / df["open"].iloc[0] - 1.0
    years = max((df.index[-1] - df.index[0]).total_seconds() / (365.2425 * 86400.0), 1e-9)
    return {
        "buy_hold_return_pct": float(r * 100),
        "buy_hold_cagr_pct": float(((1 + r) ** (1 / years) - 1) * 100) if 1 + r > 0 else -100.0,
    }


def main():
    results = []
    coverage = []
    errors = []

    for symbol in SYMBOLS:
        print(f"=== {symbol}: downloading 1h ===", flush=True)
        try:
            df1, source = fetch_symbol(symbol)
            if len(df1) < 200:
                raise RuntimeError(f"too few bars: {len(df1)}")
            df4 = to_4h(df1)
            cov = {
                "symbol": symbol,
                "source": source,
                "start": str(df1.index[0]),
                "end": str(df1.index[-1]),
                "bars_1h": int(len(df1)),
                "bars_4h": int(len(df4)),
                "missing_1h_intervals": int(max(0, round((df1.index[-1] - df1.index[0]).total_seconds() / 3600) + 1 - len(df1))),
            }
            cov.update(buy_hold(df1))
            coverage.append(cov)

            tests = [
                ("Supertrend 10,3", "4h", df4, supertrend_target(df4, 10, 3.0), 1.0, 4),
                ("Donchian 20/10", "4h", df4, donchian_target(df4, 20, 10), 1.0, 4),
                ("Donchian 20/10", "4h", df4, donchian_target(df4, 20, 10), 2.0, 4),
                ("Supertrend 40,3", "1h", df1, supertrend_target(df1, 40, 3.0), 1.0, 1),
                ("Donchian 80/40", "1h", df1, donchian_target(df1, 80, 40), 1.0, 1),
                ("Donchian 80/40", "1h", df1, donchian_target(df1, 80, 40), 2.0, 1),
            ]
            for strategy, tf, frame, tgt, lev, tfh in tests:
                m = backtest(frame, tgt, lev, tfh)
                results.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    "timeframe": tf,
                    "leverage": lev,
                    "data_start": str(frame.index[0]),
                    "data_end": str(frame.index[-1]),
                    "bars": int(len(frame)),
                    **m,
                })
                print(symbol, tf, strategy, lev, m, flush=True)
        except Exception as e:
            msg = f"{symbol}: {type(e).__name__}: {e}"
            print("ERROR", msg, flush=True)
            errors.append(msg)

    rdf = pd.DataFrame(results)
    cdf = pd.DataFrame(coverage)
    rdf.to_csv(OUT / "results.csv", index=False)
    cdf.to_csv(OUT / "coverage.csv", index=False)
    (OUT / "errors.txt").write_text("\n".join(errors), encoding="utf-8")

    if not rdf.empty:
        # Cross-asset robustness summary by fixed strategy/timeframe/leverage.
        grp = []
        for (strategy, tf, lev), g in rdf.groupby(["strategy", "timeframe", "leverage"], sort=False):
            grp.append({
                "strategy": strategy,
                "timeframe": tf,
                "leverage": lev,
                "assets": int(len(g)),
                "profitable_assets": int((g["total_return_pct"] > 0).sum()),
                "median_return_pct": float(g["total_return_pct"].median()),
                "median_cagr_pct": float(g["cagr_pct"].median()),
                "median_mdd_pct": float(g["mdd_pct"].median()),
                "median_profit_factor": float(g["profit_factor"].replace([np.inf, -np.inf], np.nan).median()),
                "median_sharpe": float(g["sharpe"].median()),
                "total_trades": int(g["trades"].sum()),
                "total_liquidations": int(g["liquidations"].sum()),
            })
        sdf = pd.DataFrame(grp)
        sdf.to_csv(OUT / "robustness_summary.csv", index=False)

        lines = []
        lines.append("# Universal crypto trend-following backtest\n")
        lines.append(f"Requested window: {START} to {END} (each asset begins at first available Binance spot data)\n")
        lines.append(f"Initial cash: {INITIAL_CASH:,.0f} KRW; fee: {FEE*100:.3f}% each side; long-only; signal on close -> next bar open; no slippage/funding/borrow cost.\n")
        lines.append("## Robustness summary\n")
        lines.append(sdf.to_markdown(index=False))
        lines.append("\n## Per-asset results\n")
        cols = ["symbol", "strategy", "timeframe", "leverage", "total_return_pct", "cagr_pct", "mdd_pct", "profit_factor", "sharpe", "win_rate_pct", "trades", "liquidations", "final_equity_krw"]
        lines.append(rdf[cols].to_markdown(index=False))
        lines.append("\n## Coverage\n")
        lines.append(cdf.to_markdown(index=False))
        if errors:
            lines.append("\n## Errors\n")
            lines.extend([f"- {e}" for e in errors])
        (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "requested_start": str(START),
        "requested_end": str(END),
        "symbols": SYMBOLS,
        "initial_cash": INITIAL_CASH,
        "fee_each_side": FEE,
        "notes": [
            "Binance spot 1h klines; 4h derived by UTC-aligned resampling.",
            "Long only. Signals are evaluated on close and executed next bar open.",
            "2x is synthetic full-account exposure; borrowing/funding/slippage excluded.",
            "If 2x intrabar equity reaches <=0 at the candle low, account is treated as liquidated.",
            "No per-asset parameter optimization.",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE", flush=True)
    if (OUT / "summary.md").exists():
        print((OUT / "summary.md").read_text(encoding="utf-8")[:20000], flush=True)


if __name__ == "__main__":
    main()
