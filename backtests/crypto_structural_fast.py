import json
from pathlib import Path
import pandas as pd

import crypto_universal_10y as base
import crypto_structural_tune as st

OUT = Path('structural_fast_output')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    datasets = []
    for symbol in base.SYMBOLS:
        print('fetch', symbol, flush=True)
        df1h, _ = base.fetch_symbol(symbol)
        tr1, oo1 = st.split_70_30(df1h)
        datasets.append((symbol, tr1, base.to_4h(tr1), oo1, base.to_4h(oo1)))

    candidates = []
    for entry_n in [160, 200]:
        for exit_n in [30, 40]:
            for ema_n in [300, 400]:
                for strong_threshold in [1.05, 1.10]:
                    for base_exposure in [0.35, 0.50]:
                        for strong_exposure in [0.75, 1.00]:
                            candidates.append({
                                'entry_n': entry_n,
                                'exit_n': exit_n,
                                'ema_n': ema_n,
                                'strong_threshold': strong_threshold,
                                'base_exposure': base_exposure,
                                'strong_exposure': strong_exposure,
                            })

    train_rows = []
    for i, cfg in enumerate(candidates, 1):
        print('train', i, '/', len(candidates), cfg, flush=True)
        s, _ = st.eval_config(datasets, cfg, 'train')
        s['train_ok'] = bool(s['profitable_assets'] >= 7 and s['median_profit_factor'] >= 1.10 and s['median_mdd_pct'] >= -45.0)
        train_rows.append(s)

    tg = pd.DataFrame(train_rows).sort_values(['train_ok','median_cagr_pct','median_return_pct'], ascending=[False,False,False])
    tg.to_csv(OUT / 'train_grid.csv', index=False)
    leaders = tg[tg.train_ok].head(16)
    if leaders.empty:
        leaders = tg.head(16)

    oos_rows, detail = [], []
    keys = ['entry_n','exit_n','ema_n','strong_threshold','base_exposure','strong_exposure']
    for rank, (_, row) in enumerate(leaders.iterrows(), 1):
        cfg = {k: (int(row[k]) if k in ['entry_n','exit_n','ema_n'] else float(row[k])) for k in keys}
        s, rows = st.eval_config(datasets, cfg, 'oos')
        s['train_rank'] = rank
        s['train_cagr_pct'] = float(row['median_cagr_pct'])
        s['train_mdd_pct'] = float(row['median_mdd_pct'])
        s['mdd35_ok'] = bool(s['profitable_assets'] >= 7 and s['median_profit_factor'] >= 1.05 and s['median_mdd_pct'] >= -35.0)
        oos_rows.append(s)
        for r in rows:
            detail.append({**s, **r})

    og = pd.DataFrame(oos_rows).sort_values(['mdd35_ok','median_cagr_pct','median_return_pct'], ascending=[False,False,False])
    og.to_csv(OUT / 'oos_leaders.csv', index=False)
    pd.DataFrame(detail).to_csv(OUT / 'oos_detail.csv', index=False)

    eligible = og[og.mdd35_ok]
    best = eligible.iloc[0].to_dict() if not eligible.empty else og.iloc[0].to_dict()
    best_mdd = og.sort_values(['median_mdd_pct','median_cagr_pct'], ascending=[False,False]).iloc[0].to_dict()
    summary = {
        'method': 'Focused 3-regime structural test: 1h Donchian state + prior completed 4h EMA/slope filter; bear=0, bull=0.35/0.50x, strong bull=0.75/1.00x; 70/30 OOS; 0.05% fee each side.',
        'candidate_count': len(candidates),
        'oos_leaders_tested': len(og),
        'best_cagr_with_mdd_le_35': best,
        'best_mdd': best_mdd,
        'previous_balanced_reference': {'return_pct': 22.575141791847088, 'mdd_pct': -26.836549972007173, 'pf': 1.3732746264067988},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('DONE')
    print(json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
