import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT = Path('turbo_500x_fast_output')
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_DAYS = 730
STEP_DAYS = 180
ENTRY_N, EXIT_N, EMA_N = 160, 30, 300
BASE_EXPOSURE, STRONG_EXPOSURE = 0.50, 0.75


def prepare(df1h):
    d4 = base.to_4h(df1h)
    ls = pd.Series(base.donchian_target(df1h, ENTRY_N, EXIT_N), index=df1h.index)
    ema = d4.close.ewm(span=EMA_N, adjust=False, min_periods=EMA_N).mean()
    bull = (d4.close > ema) & (ema > ema.shift(12))
    strong = bull & (d4.close > ema * 1.10)
    hi30 = d4.high.shift(1).rolling(180).max()
    mom24 = d4.close / d4.close.shift(6) - 1
    mom7 = d4.close / d4.close.shift(42) - 1
    atrp = pd.Series(base.wilder_atr(d4, 14), index=d4.index) / d4.close
    reg = pd.DataFrame({'ema':ema,'close4':d4.close,'bull':bull,'strong':strong,'hi30':hi30,'mom24':mom24,'mom7':mom7,'atrp':atrp}).shift(1)
    return ls, reg.reindex(df1h.index, method='ffill')


def windows(df):
    out=[]; s=df.index.min().ceil('D'); end=df.index.max().floor('D')
    while s + pd.Timedelta(days=WINDOW_DAYS) <= end:
        e=s+pd.Timedelta(days=WINDOW_DAYS)
        sub=df[(df.index>=s)&(df.index<e)]
        if len(sub)>=15000: out.append((s,e,sub.index))
        s += pd.Timedelta(days=STEP_DAYS)
    return out


def run(df, idx, ls_s, reg, cfg):
    x=df.loc[idx]
    o=x.open.to_numpy(float); lo=x.low.to_numpy(float); c=x.close.to_numpy(float)
    ls=ls_s.reindex(idx).fillna(False).to_numpy(bool)
    r=reg.reindex(idx)
    bull=r.bull.fillna(False).to_numpy(bool); strong=r.strong.fillna(False).to_numpy(bool)
    ema=r.ema.to_numpy(float); c4=r.close4.to_numpy(float); hi30=r.hi30.to_numpy(float)
    m24=r.mom24.to_numpy(float); m7=r.mom7.to_numpy(float); atrp=r.atrp.to_numpy(float)
    cash=base.INITIAL_CASH; units=0.; peak=cash; eq=np.full(len(idx),cash); liq=0; maxexp=0.
    for i in range(1,len(idx)):
        eo=cash+units*o[i]
        if eo<=0: eq[i:]=0; liq=1; break
        if eo>peak: peak=eo
        dd=eo/peak-1
        des=0.
        if ls[i-1] and bull[i-1]:
            des=BASE_EXPOSURE
            if strong[i-1]: des=STRONG_EXPOSURE
            ultra=(strong[i-1] and np.isfinite(ema[i-1]) and np.isfinite(c4[i-1]) and c4[i-1]>ema[i-1]*cfg['ultra_threshold'] and np.isfinite(hi30[i-1]) and c4[i-1]>=hi30[i-1] and np.isfinite(m24[i-1]) and m24[i-1]>0 and np.isfinite(m7[i-1]) and m7[i-1]>0)
            if ultra: des=cfg['turbo_exposure']
            if np.isfinite(atrp[i-1]) and atrp[i-1]>0.08: des=min(des,STRONG_EXPOSURE)
        if dd<=-cfg['brake_dd']: des=min(des,.50)
        if dd<=-(cfg['brake_dd']+.05): des=min(des,.25)
        if dd<=-(cfg['brake_dd']+.10): des=0.
        mult=eo/base.INITIAL_CASH
        if mult>=100: des=min(des,1.25)
        elif mult>=25: des=min(des,1.5)
        maxexp=max(maxexp,des)
        target=eo*des; cur=units*o[i]; delta=target-cur
        if abs(delta)>1e-9:
            cash -= delta + abs(delta)*base.FEE
            units += delta/o[i]
        if cash+units*lo[i] <= 0:
            eq[i:]=0; liq=1; break
        eq[i]=cash+units*c[i]
    if not liq and abs(units)>1e-12:
        cash += units*c[-1]-abs(units*c[-1])*base.FEE; units=0; eq[-1]=cash
    pk=np.maximum.accumulate(eq); ddarr=np.divide(eq,pk,out=np.ones_like(eq),where=pk>0)-1
    final=float(eq[-1])
    return {'return_pct':(final/base.INITIAL_CASH-1)*100,'multiple':final/base.INITIAL_CASH,'mdd_pct':float(ddarr.min()*100),'liquidated':liq,'max_exposure_seen':maxexp}


def main():
    data={}
    for sym in base.SYMBOLS:
        print('fetch',sym,flush=True); df,_=base.fetch_symbol(sym); ls,reg=prepare(df); data[sym]=(df,ls,reg,windows(df))
    cfgs=[{'turbo_exposure':t,'ultra_threshold':u,'brake_dd':b} for t in [1.5,2.0,2.5,3.0] for u in [1.15,1.20] for b in [0.10,0.15]]
    grid=[]; details=[]
    for k,cfg in enumerate(cfgs,1):
        print('cfg',k,'/',len(cfgs),cfg,flush=True); rows=[]
        for sym,(df,ls,reg,ws) in data.items():
            for s,e,idx in ws:
                z=run(df,idx,ls,reg,cfg); row={'symbol':sym,'start':str(s),'end':str(e),**cfg,**z}; rows.append(row); details.append(row)
        d=pd.DataFrame(rows)
        g={**cfg,'windows':len(d),'profitable_rate_pct':float((d.return_pct>0).mean()*100),'median_return_pct':float(d.return_pct.median()),'p25_return_pct':float(d.return_pct.quantile(.25)),'best_return_pct':float(d.return_pct.max()),'worst_return_pct':float(d.return_pct.min()),'median_mdd_pct':float(d.mdd_pct.median()),'worst_mdd_pct':float(d.mdd_pct.min()),'liquidations':int(d.liquidated.sum()),'windows_10x_plus':int((d.multiple>=10).sum()),'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'best_multiple':float(d.multiple.max())}
        g['score']=g['median_return_pct']+0.5*g['p25_return_pct']+0.2*g['profitable_rate_pct']+0.5*g['median_mdd_pct']-20*g['liquidations']
        grid.append(g)
    G=pd.DataFrame(grid).sort_values(['score','median_return_pct'],ascending=False); D=pd.DataFrame(details)
    G.to_csv(OUT/'grid.csv',index=False); D.to_csv(OUT/'rolling_2y_detail.csv',index=False)
    top=D.sort_values('multiple',ascending=False).head(30); top.to_csv(OUT/'top_2y_windows.csv',index=False)
    out={'method':'Fast screen: 1h Donchian160/30 + 4h EMA300 regime; 0/0.5/0.75x normal states, ultra bull 1.5-3x; 30d breakout + 24h/7d momentum + ATR<=8% + equity DD brake; 2y windows every 180d; 0.05% each side','candidate_count':len(cfgs),'best_robust':G.iloc[0].to_dict(),'best_500x_diagnostic':G.sort_values(['windows_500x_plus','best_multiple'],ascending=False).iloc[0].to_dict(),'highest_window':top.iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); print('DONE',json.dumps(out,default=str),flush=True)
if __name__=='__main__': main()
