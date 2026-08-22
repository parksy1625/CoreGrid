import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_universal_10y as base

OUT = Path('turbo_500x_output')
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 730
STEP_DAYS = 180
ENTRY_N = 160
EXIT_N = 30
EMA_N = 300
BASE_EXPOSURE = 0.50
STRONG_EXPOSURE = 0.75
FEE = base.FEE


def prepare_signals(df1h: pd.DataFrame):
    df4h = base.to_4h(df1h)
    long_state = pd.Series(base.donchian_target(df1h, ENTRY_N, EXIT_N), index=df1h.index)

    ema = df4h['close'].ewm(span=EMA_N, adjust=False, min_periods=EMA_N).mean()
    slope = ema > ema.shift(12)
    bull = (df4h['close'] > ema) & slope
    strong = bull & (df4h['close'] > ema * 1.10)

    high30 = df4h['high'].shift(1).rolling(180).max()
    mom24 = df4h['close'] / df4h['close'].shift(6) - 1.0
    mom7d = df4h['close'] / df4h['close'].shift(42) - 1.0

    atr = pd.Series(base.wilder_atr(df4h, 14), index=df4h.index)
    atr_pct = atr / df4h['close']

    frame4 = pd.DataFrame({
        'ema': ema,
        'bull': bull,
        'strong': strong,
        'high30': high30,
        'mom24': mom24,
        'mom7d': mom7d,
        'atr_pct': atr_pct,
        'close4': df4h['close'],
    }).shift(1)
    mapped = frame4.reindex(df1h.index, method='ffill')
    return long_state, mapped


def backtest_window(df, long_state, reg, cfg):
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    c = df['close'].to_numpy(float)
    ls = long_state.reindex(df.index).fillna(False).to_numpy(bool)
    rr = reg.reindex(df.index)

    bull = rr['bull'].fillna(False).to_numpy(bool)
    strong = rr['strong'].fillna(False).to_numpy(bool)
    close4 = rr['close4'].to_numpy(float)
    high30 = rr['high30'].to_numpy(float)
    mom24 = rr['mom24'].to_numpy(float)
    mom7d = rr['mom7d'].to_numpy(float)
    atrp = rr['atr_pct'].to_numpy(float)
    ema = rr['ema'].to_numpy(float)

    cash = base.INITIAL_CASH
    units = 0.0
    eq = np.full(len(df), cash, float)
    peak = cash
    liquidated = False
    turnover = 0.0
    max_exposure_seen = 0.0

    for i in range(1, len(df)):
        equity_open = cash + units * o[i]
        if equity_open <= 0:
            eq[i:] = 0.0
            liquidated = True
            break
        peak = max(peak, equity_open)
        dd = equity_open / peak - 1.0

        desired = 0.0
        if ls[i-1] and bull[i-1]:
            desired = BASE_EXPOSURE
            if strong[i-1]:
                desired = STRONG_EXPOSURE

            ultra = (
                strong[i-1]
                and np.isfinite(ema[i-1])
                and np.isfinite(close4[i-1])
                and close4[i-1] > ema[i-1] * cfg['ultra_threshold']
                and np.isfinite(high30[i-1])
                and close4[i-1] >= high30[i-1]
                and np.isfinite(mom24[i-1]) and mom24[i-1] > 0
                and np.isfinite(mom7d[i-1]) and mom7d[i-1] > 0
            )
            if ultra:
                desired = cfg['turbo_exposure']

            # Volatility governor: don't run turbo into extreme ATR expansion.
            if np.isfinite(atrp[i-1]) and atrp[i-1] > cfg['atr_cap']:
                desired = min(desired, STRONG_EXPOSURE)

        # Equity drawdown brake. Once the account retreats from its own peak,
        # aggressively cut exposure instead of allowing full turbo to persist.
        if dd <= -cfg['brake_dd']:
            desired = min(desired, 0.50)
        if dd <= -(cfg['brake_dd'] + 0.05):
            desired = min(desired, 0.25)
        if dd <= -(cfg['brake_dd'] + 0.10):
            desired = 0.0

        # Profit-protection cap after huge compounding: keep gains from being fully
        # recycled into maximum leverage. This only matters in rare explosive runs.
        multiple = equity_open / base.INITIAL_CASH
        if multiple >= 100:
            desired = min(desired, 1.25)
        elif multiple >= 25:
            desired = min(desired, 1.50)

        max_exposure_seen = max(max_exposure_seen, desired)
        target_notional = equity_open * desired
        current_notional = units * o[i]
        delta = target_notional - current_notional
        if abs(delta) > 1e-9:
            fee = abs(delta) * FEE
            cash -= delta + fee
            units += delta / o[i]
            turnover += abs(delta)

        # Simplified intrabar insolvency check. No maintenance-margin model is
        # assumed, so this is deliberately conservative only at equity <= 0.
        low_equity = cash + units * h[i] if units < 0 else cash + units * df['low'].iloc[i]
        if low_equity <= 0:
            eq[i:] = 0.0
            liquidated = True
            break
        eq[i] = cash + units * c[i]

    if not liquidated and abs(units) > 1e-12:
        fee = abs(units * c[-1]) * FEE
        cash += units * c[-1] - fee
        units = 0.0
        eq[-1] = cash

    peak_arr = np.maximum.accumulate(eq)
    dd_arr = np.divide(eq, peak_arr, out=np.ones_like(eq), where=peak_arr > 0) - 1.0
    final = float(eq[-1])
    ret = (final / base.INITIAL_CASH - 1.0) * 100.0
    return {
        'return_pct': ret,
        'multiple': final / base.INITIAL_CASH,
        'mdd_pct': float(np.nanmin(dd_arr) * 100.0),
        'liquidated': int(liquidated),
        'max_exposure_seen': max_exposure_seen,
        'turnover_x_initial': turnover / base.INITIAL_CASH,
    }


def make_windows(df):
    start = df.index.min().ceil('D')
    end = df.index.max().floor('D')
    windows = []
    s = start
    while s + pd.Timedelta(days=WINDOW_DAYS) <= end:
        e = s + pd.Timedelta(days=WINDOW_DAYS)
        sub = df[(df.index >= s) & (df.index < e)]
        if len(sub) >= 15000:
            windows.append((s, e, sub))
        s += pd.Timedelta(days=STEP_DAYS)
    return windows


def main():
    prepared = {}
    for symbol in base.SYMBOLS:
        print('fetch', symbol, flush=True)
        df, _ = base.fetch_symbol(symbol)
        ls, reg = prepare_signals(df)
        prepared[symbol] = (df, ls, reg, make_windows(df))

    candidates = []
    for turbo in [1.25, 1.50, 2.0, 2.5, 3.0]:
        for ultra in [1.15, 1.20]:
            for atr_cap in [0.06, 0.08]:
                for brake in [0.10, 0.15]:
                    candidates.append({
                        'turbo_exposure': turbo,
                        'ultra_threshold': ultra,
                        'atr_cap': atr_cap,
                        'brake_dd': brake,
                    })

    summary_rows = []
    detail_rows = []
    for ci, cfg in enumerate(candidates, 1):
        print('candidate', ci, '/', len(candidates), cfg, flush=True)
        rows = []
        for symbol, (df, ls, reg, windows) in prepared.items():
            for s, e, sub in windows:
                r = backtest_window(sub, ls, reg, cfg)
                row = {'symbol': symbol, 'start': str(s), 'end': str(e), **cfg, **r}
                rows.append(row)
                detail_rows.append(row)

        d = pd.DataFrame(rows)
        profitable = int((d['return_pct'] > 0).sum())
        total = len(d)
        r = {
            **cfg,
            'windows': total,
            'profitable_windows': profitable,
            'profitable_rate_pct': profitable / total * 100.0 if total else 0.0,
            'median_return_pct': float(d['return_pct'].median()),
            'p25_return_pct': float(d['return_pct'].quantile(0.25)),
            'worst_return_pct': float(d['return_pct'].min()),
            'best_return_pct': float(d['return_pct'].max()),
            'median_multiple': float(d['multiple'].median()),
            'best_multiple': float(d['multiple'].max()),
            'median_mdd_pct': float(d['mdd_pct'].median()),
            'worst_mdd_pct': float(d['mdd_pct'].min()),
            'liquidations': int(d['liquidated'].sum()),
            'windows_10x_plus': int((d['multiple'] >= 10).sum()),
            'windows_100x_plus': int((d['multiple'] >= 100).sum()),
            'windows_500x_plus': int((d['multiple'] >= 500).sum()),
        }
        # Robust ranking: reward median / p25 returns, but apply large penalties to
        # liquidation and severe typical drawdown. Best-window return is diagnostic,
        # not the dominant optimization target.
        r['score'] = (
            math.log1p(max(r['median_return_pct'], -99.0) / 100.0) * 100.0
            + math.log1p(max(r['p25_return_pct'], -99.0) / 100.0) * 60.0
            + r['profitable_rate_pct'] * 0.25
            + max(r['median_mdd_pct'], -100.0) * 0.50
            - r['liquidations'] * 20.0
        )
        summary_rows.append(r)

    grid = pd.DataFrame(summary_rows).sort_values(['score', 'median_return_pct'], ascending=False)
    grid.to_csv(OUT / 'grid.csv', index=False)
    details = pd.DataFrame(detail_rows)
    details.to_csv(OUT / 'rolling_2y_detail.csv', index=False)

    best_robust = grid.iloc[0].to_dict()
    best_500 = grid.sort_values(['windows_500x_plus', 'best_multiple', 'median_return_pct'], ascending=False).iloc[0].to_dict()

    # Best individual 2-year windows across all configurations, explicitly marked
    # as diagnostic because selecting them after seeing outcomes is not predictive.
    top_windows = details.sort_values('multiple', ascending=False).head(50)
    top_windows.to_csv(OUT / 'top_2y_windows.csv', index=False)

    out = {
        'method': '1h Donchian 160/30 + prior-completed 4h EMA300 regime; bear 0x, bull 0.5x, strong 0.75x, ultra bull 1.25-3x; momentum + 30d breakout + ATR governor + equity drawdown brake; 2-year windows stepped every 180d; fee 0.05% each side',
        'candidate_count': len(candidates),
        'best_robust_configuration': best_robust,
        'best_500x_diagnostic_configuration': best_500,
        'highest_2y_window': top_windows.iloc[0].to_dict() if len(top_windows) else None,
        'warning': 'Highest-window and 500x counts are in-sample diagnostics across many overlapping windows and configurations, not a forward-return guarantee.',
    }
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('DONE')
    print(json.dumps(out, default=str), flush=True)


if __name__ == '__main__':
    main()
