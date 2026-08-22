import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent))
import crypto_universal_10y as base

OUT=Path('mdd_tuning_output'); OUT.mkdir(exist_ok=True)

def summarize(rows):
    d=pd.DataFrame(rows)
    return {
        'assets':len(d),
        'profitable_assets':int((d.total_return_pct>0).sum()),
        'median_return_pct':float(d.total_return_pct.median()),
        'median_cagr_pct':float(d.cagr_pct.median()),
        'median_mdd_pct':float(d.mdd_pct.median()),
        'median_profit_factor':float(d.profit_factor.replace([np.inf,-np.inf],np.nan).median()),
        'median_sharpe':float(d.sharpe.median()),
        'total_trades':int(d.trades.sum()),
        'total_liquidations':int(d.liquidations.sum()),
    }

def ema_filter_target(df, entry_n, exit_n, ema_n):
    dc=base.donchian_target(df,entry_n,exit_n)
    ema=df.close.ewm(span=ema_n,adjust=False).mean().to_numpy()
    close=df.close.to_numpy()
    # No new/continued long exposure below long-term EMA.
    return dc & (close>ema)

def eval_cfg(datasets, entry_n, exit_n, ema_n, lev, part):
    rows=[]
    for sym,df in datasets:
        cut=int(len(df)*0.70)
        x=df.iloc[:cut].copy() if part=='train' else df.iloc[cut:].copy()
        if len(x)<500: continue
        target=ema_filter_target(x,entry_n,exit_n,ema_n)
        r=base.backtest(x,target,lev,1)
        r['symbol']=sym; rows.append(r)
    s=summarize(rows)
    s.update({'entry_n':entry_n,'exit_n':exit_n,'ema_n':ema_n,'leverage':lev,'part':part})
    return s,rows

def main():
    datasets=[]
    for sym in base.SYMBOLS:
        print('fetch',sym,flush=True)
        df,_=base.fetch_symbol(sym)
        datasets.append((sym,df))

    entries=[80,120,160,200]
    exits=[30,40,60,80]
    emas=[100,200,300,400]
    leverages=[0.50,0.65,0.80,1.00]

    train=[]
    for e in entries:
      for x in exits:
        if x>=e: continue
        for ema in emas:
          for lev in leverages:
            s,_=eval_cfg(datasets,e,x,ema,lev,'train')
            # Require usable edge before preferring lower MDD.
            s['eligible']=bool(s['profitable_assets']>=8 and s['median_return_pct']>100 and s['median_profit_factor']>=1.10 and s['total_liquidations']==0)
            train.append(s)
            print('train',e,x,ema,lev,s['median_return_pct'],s['median_mdd_pct'],flush=True)

    t=pd.DataFrame(train)
    # Among viable configs, minimize drawdown first, then maximize CAGR/PF.
    t=t.sort_values(['eligible','median_mdd_pct','median_cagr_pct','median_profit_factor'],ascending=[False,False,False,False])
    t.to_csv(OUT/'train_grid.csv',index=False)

    leaders=t[t.eligible].head(20)
    if leaders.empty: leaders=t.head(20)
    oos=[]; detail=[]
    for rank,(_,r) in enumerate(leaders.iterrows(),1):
        s,rows=eval_cfg(datasets,int(r.entry_n),int(r.exit_n),int(r.ema_n),float(r.leverage),'oos')
        s['train_rank']=rank
        s['train_return_pct']=float(r.median_return_pct)
        s['train_mdd_pct']=float(r.median_mdd_pct)
        s['robust_oos']=bool(s['profitable_assets']>=6 and s['median_return_pct']>0 and s['median_profit_factor']>=1.0 and s['total_liquidations']==0)
        oos.append(s)
        for rr in rows: detail.append({**s,**rr})

    o=pd.DataFrame(oos).sort_values(['robust_oos','median_mdd_pct','median_cagr_pct'],ascending=[False,False,False])
    o.to_csv(OUT/'oos_leaders.csv',index=False)
    pd.DataFrame(detail).to_csv(OUT/'oos_detail.csv',index=False)

    best=o.iloc[0].to_dict()
    # Best positive-return OOS by lowest MDD, even if strict robust rule fails.
    pos=o[o.median_return_pct>0]
    best_positive=(pos.sort_values(['median_mdd_pct','median_cagr_pct'],ascending=[False,False]).iloc[0].to_dict() if len(pos) else None)
    summary={
      'method':'10 crypto, 1h Donchian + EMA regime filter, 70/30 train/OOS, MDD-first objective, exposure 0.5-1.0x, 0.05% fee each side, next-bar open',
      'candidate_count':int(len(t)),
      'best_oos_mdd_first':best,
      'best_positive_oos_mdd_first':best_positive,
      'baseline_reference':{
        '1h_donchian_80_40_1x_full_period_mdd_pct':-69.80494820638808,
        '1h_donchian_160_80_1x_full_period_mdd_pct':-67.6816983202788,
        'previous_walkforward_best_oos_return_pct':2.266611702048798,
        'previous_walkforward_best_oos_mdd_pct':-72.06757151891688
      }
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print('DONE',json.dumps(best,default=str),flush=True)

if __name__=='__main__': main()
