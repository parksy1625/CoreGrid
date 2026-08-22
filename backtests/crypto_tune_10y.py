import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_universal_10y as base

OUT = Path("tuning_output")
OUT.mkdir(parents=True, exist_ok=True)


def summarize(rows):
    d = pd.DataFrame(rows)
    return {
        "assets": int(len(d)),
        "profitable_assets": int((d["total_return_pct"] > 0).sum()),
        "median_return_pct": float(d["total_return_pct"].median()),
        "median_cagr_pct": float(d["cagr_pct"].median()),
        "median_mdd_pct": float(d["mdd_pct"].median()),
        "median_profit_factor": float(d["profit_factor"].replace([np.inf, -np.inf], np.nan).median()),
        "median_sharpe": float(d["sharpe"].median()),
        "total_trades": int(d["trades"].sum()),
        "total_liquidations": int(d["liquidations"].sum()),
    }


def evaluate(name, timeframe, params, leverage, datasets):
    rows = []
    for symbol, df1h, df4h in datasets:
        df = df4h if timeframe == "4h" else df1h
        if name == "donchian":
            target = base.donchian_target(df, int(params["entry_n"]), int(params["exit_n"]))
        elif name == "supertrend":
            target = base.supertrend_target(df, int(params["period"]), float(params["mult"]))
        else:
            raise ValueError(name)
        r = base.backtest(df, target, float(leverage), 4 if timeframe == "4h" else 1)
        r["symbol"] = symbol
        rows.append(r)
    s = summarize(rows)
    s.update({
        "family": name,
        "timeframe": timeframe,
        "leverage": float(leverage),
        "params": json.dumps(params, sort_keys=True),
    })
    return s, rows


def main():
    datasets = []
    for symbol in base.SYMBOLS:
        print("fetch", symbol, flush=True)
        df1h, _ = base.fetch_symbol(symbol)
        datasets.append((symbol, df1h, base.to_4h(df1h)))

    candidates = []

    # 4H Donchian: broaden around the proven 20/10 baseline.
    entries_4h = [10, 12, 16, 20, 24, 30, 40, 55]
    exits_4h = [4, 5, 8, 10, 12, 16, 20, 28]
    for e in entries_4h:
        for x in exits_4h:
            if x >= e:
                continue
            candidates.append(("donchian", "4h", {"entry_n": e, "exit_n": x}))

    # 4H Supertrend: modest grid to avoid excessive overfitting.
    for p in [7, 10, 14, 20, 30]:
        for m in [1.8, 2.2, 2.6, 3.0, 3.5]:
            candidates.append(("supertrend", "4h", {"period": p, "mult": m}))

    # 1H Donchian: same general time horizon as the 4H system plus faster/slower variants.
    for e in [48, 64, 80, 96, 120, 160]:
        for ratio in [0.35, 0.50, 0.65]:
            x = max(4, int(round(e * ratio)))
            if x < e:
                candidates.append(("donchian", "1h", {"entry_n": e, "exit_n": x}))

    grid = []
    detail_rows = []
    for idx, (fam, tf, params) in enumerate(candidates, 1):
        print(f"grid {idx}/{len(candidates)} {fam} {tf} {params}", flush=True)
        s, rows = evaluate(fam, tf, params, 1.0, datasets)
        grid.append(s)
        for r in rows:
            detail_rows.append({**s, **r})

    g = pd.DataFrame(grid)
    # Two leaderboards: pure return and robust return. Robust requires broad profitability,
    # no severe median drawdown and a meaningful PF edge after fees.
    g["robust_ok"] = (
        (g["profitable_assets"] >= 8)
        & (g["median_profit_factor"] >= 1.10)
        & (g["median_mdd_pct"] >= -85.0)
        & (g["total_liquidations"] == 0)
    )
    g = g.sort_values(["robust_ok", "median_cagr_pct", "median_return_pct"], ascending=[False, False, False])
    g.to_csv(OUT / "parameter_grid.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT / "parameter_detail.csv", index=False)

    robust_top = g[g["robust_ok"]].head(8)
    if robust_top.empty:
        robust_top = g.head(8)

    lev_rows = []
    lev_detail = []
    leverages = [1.0, 1.15, 1.25, 1.40, 1.60, 1.80, 2.0]
    for _, row in robust_top.iterrows():
        params = json.loads(row["params"])
        for lev in leverages:
            print("leverage", row["family"], row["timeframe"], params, lev, flush=True)
            s, rows = evaluate(row["family"], row["timeframe"], params, lev, datasets)
            s["robust_ok"] = bool(
                s["profitable_assets"] >= 8
                and s["median_profit_factor"] >= 1.05
                and s["median_mdd_pct"] >= -90.0
                and s["total_liquidations"] == 0
            )
            lev_rows.append(s)
            for r in rows:
                lev_detail.append({**s, **r})

    l = pd.DataFrame(lev_rows)
    l = l.sort_values(["robust_ok", "median_cagr_pct", "median_return_pct"], ascending=[False, False, False])
    l.to_csv(OUT / "leverage_grid.csv", index=False)
    pd.DataFrame(lev_detail).to_csv(OUT / "leverage_detail.csv", index=False)

    best_robust = l[l["robust_ok"]].iloc[0].to_dict() if (l["robust_ok"].any()) else l.iloc[0].to_dict()
    best_raw = l.sort_values("median_return_pct", ascending=False).iloc[0].to_dict()
    best_param_1x = g.iloc[0].to_dict()

    baseline = {
        "4h_donchian_20_10_1x": {
            "median_return_pct": 1392.471358694845,
            "median_cagr_pct": 41.598380688014004,
            "median_mdd_pct": -74.49083579790053,
            "median_profit_factor": 1.173017184969805,
            "profitable_assets": 9,
        },
        "4h_supertrend_10_3_1x": {
            "median_return_pct": 1670.2056653364957,
            "median_cagr_pct": 39.99532065836195,
            "median_mdd_pct": -79.01973994925092,
            "median_profit_factor": 1.1726222577716428,
            "profitable_assets": 9,
        },
        "1h_donchian_80_40_1x": {
            "median_return_pct": 1196.5370076339755,
            "median_cagr_pct": 34.54219481088175,
            "median_mdd_pct": -69.80494820638808,
            "median_profit_factor": 1.2383011711280028,
            "profitable_assets": 9,
        },
    }

    summary = {
        "method": "same parameters across 10 crypto assets; 10-year/listing-to-2026-08-22; long-only; 0.05% fee each side; next-bar open execution",
        "candidate_count_1x": int(len(g)),
        "leverage_refinement_rows": int(len(l)),
        "best_parameter_1x": best_param_1x,
        "best_robust_with_leverage": best_robust,
        "best_raw_return_with_leverage": best_raw,
        "baseline": baseline,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def fmt(d):
        return (
            f"{d['family']} {d['timeframe']} {d['params']} lev={d['leverage']} | "
            f"profitable={int(d['profitable_assets'])}/10 | median return={d['median_return_pct']:.2f}% | "
            f"CAGR={d['median_cagr_pct']:.2f}% | MDD={d['median_mdd_pct']:.2f}% | "
            f"PF={d['median_profit_factor']:.3f} | Sharpe={d['median_sharpe']:.3f}"
        )

    md = [
        "# Crypto strategy tuning — 10-year robustness grid",
        "",
        "Optimization goal: increase cross-asset return without accepting a one-coin overfit. The same parameters are used on all 10 assets.",
        "",
        "## Best robust configuration",
        "",
        fmt(best_robust),
        "",
        "## Highest raw median-return configuration",
        "",
        fmt(best_raw),
        "",
        "## Best 1x parameter configuration",
        "",
        fmt(best_param_1x),
        "",
        "## Robust constraint",
        "",
        "At least 8/10 profitable assets, median PF >= 1.05 after leverage refinement, median MDD >= -90%, and zero liquidations.",
        "",
        "See parameter_grid.csv, leverage_grid.csv, and detail files for the complete results.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("DONE")
    print(fmt(best_robust))


if __name__ == "__main__":
    main()
