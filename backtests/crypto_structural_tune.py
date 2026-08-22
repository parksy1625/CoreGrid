import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_universal_10y as base

OUT = Path('structural_tuning_output')
OUT.mkdir(parents=True, exist_ok=True)


def metrics_from_equity(df, eq, trades, timeframe_hours=1):
    eq = np.asarray(eq, float)
    final_equity = float(eq[-1])
    total_return = (final_equity / base.INITIAL_CASH - 1.0) * 100.0
    years = max((df.index[-1] - df.index[0]).total_seconds() / (365.2425 * 86400.0), 1e-9)
    cagr = ((final_equity / base.INITIAL_CASH) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0
    peak = np.maximum.accumulate(eq)
    dd = np.divide(eq, peak, out=np.ones_like(eq), where=peak != 0) - 1.0
    mdd = float(np.nanmin(dd) * 100.0)
    rets = np.zeros(len(eq) - 1)
    ok = eq[:-1] > 0
    rets[ok] = eq[1:][ok] / eq[:-1][ok] - 1.0
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0
    if len(rets) > 1 and np.std(rets, ddof=1) > 0:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(365.2425 * 24.0 / timeframe_hours))
    pnls = np.asarray([t['pnl'] for t in trades], float)
    gp = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
    gl = float(-pnls[pnls < 0].sum()) if len(pnls) else 0.0
    pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0.0)
    wins = int((pnls > 0).sum()) if len(pnls) else 0
    return {
        'total_return_pct': total_return,
        'cagr_pct': cagr,
        'mdd_pct': mdd,
        'profit_factor': pf,
        'sharpe': sharpe,
        'win_rate_pct': wins / len(pnls) * 100.0 if len(pnls) else 0.0,
        'trades': int(len(trades)),
        'liquidations': 0,
        'final_equity_krw': final_equity,
    }


def adaptive_target(df1h, df4h, entry_n, exit_n, ema_n, strong_threshold, base_exposure, strong_exposure):
    # 1h Donchian state.
    long_state = base.donchian_target(df1h, entry_n, exit_n)

    # 4h regime filter, shifted by one completed 4h bar to avoid look-ahead when mapped to 1h.
    ema = df4h['close'].ewm(span=ema_n, adjust=False, min_periods=ema_n).mean()
    slope = ema > ema.shift(12)
    bull = (df4h['close'] > ema) & slope
    strong = bull & (df4h['close'] > ema * strong_threshold)

    regime = pd.Series(0.0, index=df4h.index)
    regime[bull] = base_exposure
    regime[strong] = strong_exposure
    regime = regime.shift(1).reindex(df1h.index, method='ffill').fillna(0.0).to_numpy(float)

    exposure = np.where(long_state, regime, 0.0)
    return exposure


def backtest_variable_exposure(df, exposure):
    o = df['open'].to_numpy(float)
    c = df['close'].to_numpy(float)
    n = len(df)
    eq = np.full(n, base.INITIAL_CASH, dtype=float)
    cash = base.INITIAL_CASH
    units = 0.0
    current_exposure = 0.0
    trade_start_equity = None
    trade_start_idx = None
    trades = []

    for i in range(1, n):
        prev_equity = cash + units * o[i]
        desired = float(exposure[i - 1])
        desired = max(0.0, min(desired, 1.0))
        target_notional = prev_equity * desired
        current_notional = units * o[i]
        delta = target_notional - current_notional

        # Treat transitions into/out of the strategy as trades. Rebalancing within a trade is allowed.
        if current_exposure <= 1e-12 and desired > 1e-12:
            trade_start_equity = prev_equity
            trade_start_idx = i
        if abs(delta) > 1e-9:
            fee = abs(delta) * base.FEE
            cash -= delta + fee
            units += delta / o[i]

        if current_exposure > 1e-12 and desired <= 1e-12:
            end_equity = cash + units * o[i]
            # units should be ~0 after rebalance.
            pnl = end_equity - (trade_start_equity if trade_start_equity is not None else end_equity)
            trades.append({'entry_i': trade_start_idx, 'exit_i': i, 'pnl': pnl})
            trade_start_equity = None
            trade_start_idx = None

        current_exposure = desired
        eq[i] = cash + units * c[i]

    # Close residual at final close, including exit fee.
    if abs(units) > 1e-12:
        notional = abs(units * c[-1])
        fee = notional * base.FEE
        cash += units * c[-1] - fee
        units = 0.0
        eq[-1] = cash
        if trade_start_equity is not None:
            trades.append({'entry_i': trade_start_idx, 'exit_i': n - 1, 'pnl': cash - trade_start_equity})

    return metrics_from_equity(df, eq, trades, 1)


def summarize(rows):
    d = pd.DataFrame(rows)
    return {
        'assets': int(len(d)),
        'profitable_assets': int((d.total_return_pct > 0).sum()),
        'median_return_pct': float(d.total_return_pct.median()),
        'median_cagr_pct': float(d.cagr_pct.median()),
        'median_mdd_pct': float(d.mdd_pct.median()),
        'median_profit_factor': float(d.profit_factor.replace([np.inf, -np.inf], np.nan).median()),
        'median_sharpe': float(d.sharpe.median()),
        'total_trades': int(d.trades.sum()),
    }


def split_70_30(df):
    cut = int(len(df) * 0.70)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def eval_config(datasets, cfg, part):
    rows = []
    for symbol, train1h, train4h, oos1h, oos4h in datasets:
        df1h, df4h = (train1h, train4h) if part == 'train' else (oos1h, oos4h)
        if len(df1h) < max(cfg['entry_n'], cfg['ema_n'] * 4) + 100:
            continue
        exp = adaptive_target(df1h, df4h, **cfg)
        r = backtest_variable_exposure(df1h, exp)
        r['symbol'] = symbol
        rows.append(r)
    s = summarize(rows)
    s.update(cfg)
    s['part'] = part
    return s, rows


def main():
    datasets = []
    for symbol in base.SYMBOLS:
        print('fetch', symbol, flush=True)
        df1h, _ = base.fetch_symbol(symbol)
        tr1, oo1 = split_70_30(df1h)
        tr4 = base.to_4h(tr1)
        oo4 = base.to_4h(oo1)
        datasets.append((symbol, tr1, tr4, oo1, oo4))

    candidates = []
    for entry_n in [120, 160, 200]:
        for exit_n in [20, 30, 40, 60]:
            if exit_n >= entry_n:
                continue
            for ema_n in [200, 300, 400]:
                for strong_threshold in [1.05, 1.10]:
                    for base_exposure in [0.35, 0.50]:
                        for strong_exposure in [0.75, 1.00]:
                            if strong_exposure <= base_exposure:
                                continue
                            candidates.append({
                                'entry_n': entry_n, 'exit_n': exit_n, 'ema_n': ema_n,
                                'strong_threshold': strong_threshold,
                                'base_exposure': base_exposure, 'strong_exposure': strong_exposure,
                            })

    train_grid = []
    for i, cfg in enumerate(candidates, 1):
        print('train', i, '/', len(candidates), cfg, flush=True)
        s, _ = eval_config(datasets, cfg, 'train')
        s['train_ok'] = bool(
            s['profitable_assets'] >= 7
            and s['median_profit_factor'] >= 1.10
            and s['median_mdd_pct'] >= -40.0
        )
        train_grid.append(s)

    tg = pd.DataFrame(train_grid)
    tg = tg.sort_values(['train_ok', 'median_cagr_pct', 'median_return_pct'], ascending=[False, False, False])
    tg.to_csv(OUT / 'train_grid.csv', index=False)

    leaders = tg[tg.train_ok].head(24)
    if leaders.empty:
        leaders = tg.head(24)

    oos_rows = []
    detail = []
    for rank, (_, row) in enumerate(leaders.iterrows(), 1):
        cfg = {k: (int(row[k]) if k in ['entry_n','exit_n','ema_n'] else float(row[k])) for k in ['entry_n','exit_n','ema_n','strong_threshold','base_exposure','strong_exposure']}
        s, rows = eval_config(datasets, cfg, 'oos')
        s['train_rank'] = rank
        s['train_median_cagr_pct'] = float(row['median_cagr_pct'])
        s['train_median_mdd_pct'] = float(row['median_mdd_pct'])
        s['robust_oos'] = bool(
            s['profitable_assets'] >= 7
            and s['median_profit_factor'] >= 1.05
            and s['median_mdd_pct'] >= -30.0
        )
        oos_rows.append(s)
        for r in rows:
            detail.append({**s, **r})

    og = pd.DataFrame(oos_rows)
    og = og.sort_values(['robust_oos','median_cagr_pct','median_return_pct'], ascending=[False,False,False])
    og.to_csv(OUT / 'oos_leaders.csv', index=False)
    pd.DataFrame(detail).to_csv(OUT / 'oos_detail.csv', index=False)

    clean = og[og.robust_oos]
    best_clean = clean.iloc[0].to_dict() if not clean.empty else None
    best_mdd = og.sort_values(['median_mdd_pct','median_cagr_pct'], ascending=[False,False]).iloc[0].to_dict()
    best_return_under_35 = og[og.median_mdd_pct >= -35.0].sort_values('median_cagr_pct', ascending=False)
    best_return_under_35 = best_return_under_35.iloc[0].to_dict() if not best_return_under_35.empty else None

    summary = {
        'method': '1h Donchian execution + 4h EMA/slope regime + adaptive 0.35/0.5 base exposure and 0.75/1.0 strong-trend exposure; 70/30 train/OOS; fee 0.05% each side; next-bar open',
        'candidate_count': len(candidates),
        'oos_leaders_tested': len(og),
        'best_clean_oos_mdd_le_30': best_clean,
        'best_oos_mdd': best_mdd,
        'best_oos_cagr_with_mdd_le_35': best_return_under_35,
        'reference_previous': {
            'mdd_focused_200_40_ema400_0.5_oos_return_pct': 12.47726492540523,
            'mdd_focused_200_40_ema400_0.5_oos_mdd_pct': -23.607494308111555,
            'balanced_160_30_ema400_0.5_oos_return_pct': 22.575141791847088,
            'balanced_160_30_ema400_0.5_oos_mdd_pct': -26.836549972007173,
        }
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('DONE', json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
