import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_universal_10y as base

OUT = Path("walkforward_output")
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


def evaluate(params, leverage, datasets):
    rows = []
    for symbol, df in datasets:
        target = base.donchian_target(df, int(params["entry_n"]), int(params["exit_n"]))
        r = base.backtest(df, target, float(leverage), 1)
        r["symbol"] = symbol
        rows.append(r)
    return summarize(rows), rows


def split_dataset(df):
    cut = max(500, int(len(df) * 0.70))
    # Keep enough pre-roll before OOS start so rolling Donchian levels are warmed up.
    warmup = 300
    train = df.iloc[:cut].copy()
    test = df.iloc[max(0, cut - warmup):].copy()
    test.attrs["oos_start"] = df.index[cut]
    return train, test


def robust(s, max_mdd=-85.0, min_pf=1.10, min_profitable=8):
    return bool(
        s["profitable_assets"] >= min_profitable
        and s["median_profit_factor"] >= min_pf
        and s["median_mdd_pct"] >= max_mdd
        and s["total_liquidations"] == 0
    )


def main():
    train_sets = []
    test_sets = []
    split_rows = []

    for symbol in base.SYMBOLS:
        print("fetch", symbol, flush=True)
        df, _ = base.fetch_symbol(symbol)
        tr, te = split_dataset(df)
        train_sets.append((symbol, tr))
        test_sets.append((symbol, te))
        split_rows.append({
            "symbol": symbol,
            "full_start": str(df.index[0]),
            "split_time": str(te.attrs["oos_start"]),
            "full_end": str(df.index[-1]),
            "train_bars": len(tr),
            "test_bars_with_warmup": len(te),
        })
    pd.DataFrame(split_rows).to_csv(OUT / "splits.csv", index=False)

    # Fine grid around the successful 160/80 region.
    entries = list(range(120, 221, 10))
    exits = list(range(40, 121, 10))
    candidates = [(e, x) for e in entries for x in exits if x < e]

    param_rows = []
    param_detail = []
    for i, (e, x) in enumerate(candidates, 1):
        params = {"entry_n": e, "exit_n": x}
        print(f"param {i}/{len(candidates)} {params}", flush=True)
        s, rows = evaluate(params, 1.0, train_sets)
        rec = {
            **s,
            "entry_n": e,
            "exit_n": x,
            "leverage": 1.0,
            "robust_train": robust(s),
        }
        param_rows.append(rec)
        for r in rows:
            param_detail.append({**rec, **r})

    pg = pd.DataFrame(param_rows)
    pg = pg.sort_values(
        ["robust_train", "median_cagr_pct", "median_return_pct", "median_mdd_pct"],
        ascending=[False, False, False, False],
    )
    pg.to_csv(OUT / "train_parameter_grid.csv", index=False)
    pd.DataFrame(param_detail).to_csv(OUT / "train_parameter_detail.csv", index=False)

    top_params = pg[pg["robust_train"]].head(12)
    if top_params.empty:
        top_params = pg.head(12)

    leverages = [1.0, 1.10, 1.20, 1.25, 1.30, 1.40, 1.50]
    train_lev_rows = []
    for _, p in top_params.iterrows():
        params = {"entry_n": int(p.entry_n), "exit_n": int(p.exit_n)}
        for lev in leverages:
            print("train leverage", params, lev, flush=True)
            s, _ = evaluate(params, lev, train_sets)
            train_lev_rows.append({
                **s,
                **params,
                "leverage": lev,
                "robust_train": robust(s, max_mdd=-88.0, min_pf=1.05),
            })

    tl = pd.DataFrame(train_lev_rows).sort_values(
        ["robust_train", "median_cagr_pct", "median_return_pct", "median_mdd_pct"],
        ascending=[False, False, False, False],
    )
    tl.to_csv(OUT / "train_leverage_grid.csv", index=False)

    # OOS is never used to generate parameter combinations. Evaluate only train-selected leaders.
    leaders = tl[tl["robust_train"]].head(15)
    if leaders.empty:
        leaders = tl.head(15)

    oos_rows = []
    oos_detail = []
    for rank, (_, p) in enumerate(leaders.iterrows(), 1):
        params = {"entry_n": int(p.entry_n), "exit_n": int(p.exit_n)}
        lev = float(p.leverage)
        print("OOS", rank, params, lev, flush=True)
        s, rows = evaluate(params, lev, test_sets)
        rec = {
            **s,
            **params,
            "leverage": lev,
            "train_rank": rank,
            "train_median_return_pct": float(p.median_return_pct),
            "train_median_cagr_pct": float(p.median_cagr_pct),
            "train_median_mdd_pct": float(p.median_mdd_pct),
            "train_median_profit_factor": float(p.median_profit_factor),
            "robust_oos": robust(s, max_mdd=-90.0, min_pf=1.00, min_profitable=7),
        }
        oos_rows.append(rec)
        for r in rows:
            oos_detail.append({**rec, **r})

    oos = pd.DataFrame(oos_rows)
    oos.to_csv(OUT / "oos_leaders.csv", index=False)
    pd.DataFrame(oos_detail).to_csv(OUT / "oos_detail.csv", index=False)

    # Primary choice is strictly the #1 train-selected configuration; OOS is validation only.
    chosen = oos.sort_values("train_rank").iloc[0].to_dict()
    # Also expose best OOS result as diagnostic, explicitly not a clean validation pick.
    best_oos_diag = oos.sort_values(
        ["robust_oos", "median_cagr_pct", "median_return_pct"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    summary = {
        "method": "10 crypto, 1h Donchian fine grid; per-asset chronological 70% train / 30% OOS; long-only; 0.05% each side; next-bar open; same parameters across all assets",
        "parameter_candidates": len(candidates),
        "train_leverage_candidates": len(tl),
        "oos_leaders_tested": len(oos),
        "train_selected_configuration_oos_result": chosen,
        "best_oos_diagnostic_not_clean_selection": best_oos_diag,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def f(d):
        return (
            f"Donchian {int(d['entry_n'])}/{int(d['exit_n'])} 1H x{float(d['leverage']):.2f} | "
            f"profitable={int(d['profitable_assets'])}/10 | median return={float(d['median_return_pct']):.2f}% | "
            f"CAGR={float(d['median_cagr_pct']):.2f}% | MDD={float(d['median_mdd_pct']):.2f}% | "
            f"PF={float(d['median_profit_factor']):.3f} | Sharpe={float(d['median_sharpe']):.3f}"
        )

    md = [
        "# Walk-forward crypto Donchian fine tuning",
        "",
        "Parameters are selected using only the first 70% of each coin's history. The final 30% is used as out-of-sample validation.",
        "",
        "## Train-selected #1, then evaluated OOS",
        "",
        f(chosen),
        "",
        "## Best OOS diagnostic (not a clean validation pick)",
        "",
        f(best_oos_diag),
        "",
        "See train_parameter_grid.csv, train_leverage_grid.csv, oos_leaders.csv and oos_detail.csv.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("DONE")
    print("TRAIN SELECTED OOS:", f(chosen))


if __name__ == "__main__":
    main()
