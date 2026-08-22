import json
from pathlib import Path
import numpy as np, pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v2_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS
FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    mom24=c/c.shift(24)-1; mom7=c/c.shift(24*7)-1
    hh30=df.high.rolling(24*30).max().shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/24,adjust=False,min_periods=24).mean()/c
    score=(c/c.shift(24*30)-1)*0.5+(c/c.shift(24*7)-1)*0.3+(c/c.shift(24)-1)*0.2
    bull=(c>ema)&(ema>ema.shift(24*3))
    ultra=bull&(c>ema*1.12)&(c>hh30)&(mom24>0)&(mom7>0)&(atr<0.08)
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'bull':bull.astype(int),'ultra':ultra.astype(int),'atr':atr})

def run(panel,cfg,start,end):
    dates=panel.index[(panel.index>=start)&(panel.index<=end)]
    eq=1.0; peak=1.0; maxdd=0.0; cur=None; entry=0; exp=0.0; liquid=False
    for k in range(1,len(dates)):
        t0,t1=dates[k-1],dates[k]
        rows=panel.loc[t0]
        valid=rows[(rows.bull>0)&np.isfinite(rows.score)]
        chosen=None if len(valid)==0 else valid.score.idxmax()
        dd=eq/peak-1
        brake=cfg['brake'] if dd<=-cfg['dd'] else 1.0
        desired=0.0
        if chosen is not None:
            r=rows.loc[chosen]
            desired=0.5
            if r.ultra>0: desired=cfg['turbo']
            if cur==chosen and entry>0:
                gain=panel.loc[t0,(chosen,'close')]/entry-1
                if gain>0.12: desired=max(desired,cfg['pyr1'])
                if gain>0.25: desired=max(desired,cfg['pyr2'])
            desired*=brake
        # turnover fee on exposure/symbol changes
        turn=0.0
        if cur!=chosen: turn=exp+desired
        else: turn=abs(desired-exp)
        eq*=max(0.0,1-turn*FEE)
        if cur is not None and exp>0:
            p0=panel.loc[t0,(cur,'close')]; p1=panel.loc[t1,(cur,'close')]
            ret=p1/p0-1
            eq*=1+exp*ret
            if eq<=0: eq=0; liquid=True; break
        if chosen!=cur:
            cur=chosen; entry=0 if chosen is None else panel.loc[t1,(chosen,'open')]
        exp=desired
        peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
    return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':int(liquid)}

def main():
    ps=[]
    for s in SYMS:
        print('fetch',s,flush=True)
        d,_=base.fetch_symbol(s); p=prep(d); p.columns=pd.MultiIndex.from_product([[s],p.columns]); ps.append(p)
    panel=pd.concat(ps,axis=1).sort_index()
    # common hourly timeline with per-asset NaNs allowed
    cfgs=[]
    for turbo in [1.5,2.0,2.5,3.0]:
      for p1,p2 in [(1.0,1.5),(1.25,2.0)]:
       for dd,br in [(0.10,0.5),(0.15,0.35)]: cfgs.append({'turbo':turbo,'pyr1':p1,'pyr2':p2,'dd':dd,'brake':br})
    starts=pd.date_range(max(panel.index.min(),pd.Timestamp('2018-01-01',tz='UTC')), panel.index.max()-pd.Timedelta(days=730), freq='180D')
    rows=[]
    for ci,cfg in enumerate(cfgs,1):
        vals=[]
        print('cfg',ci,len(cfgs),cfg,flush=True)
        for st in starts:
            en=st+pd.Timedelta(days=730)
            r=run(panel,cfg,st,en); vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        cfg['windows']=len(d); cfg['median_return_pct']=d.return_pct.median(); cfg['best_return_pct']=d.return_pct.max(); cfg['worst_return_pct']=d.return_pct.min(); cfg['median_mdd_pct']=d.mdd_pct.median(); cfg['worst_mdd_pct']=d.mdd_pct.min(); cfg['profitable_rate_pct']=(d.return_pct>0).mean()*100; cfg['windows_10x_plus']=(d.multiple>=10).sum(); cfg['windows_100x_plus']=(d.multiple>=100).sum(); cfg['windows_500x_plus']=(d.multiple>=500).sum(); cfg['liquidations']=d.liquidated.sum()
    g=pd.DataFrame(cfgs)
    g['score']=g.median_return_pct-0.7*g.worst_mdd_pct.abs()+2*g.windows_10x_plus+20*g.windows_100x_plus
    g=g.sort_values(['windows_500x_plus','windows_100x_plus','windows_10x_plus','score'],ascending=False)
    pd.DataFrame(rows).to_csv(OUT/'rolling_2y.csv',index=False); g.to_csv(OUT/'grid.csv',index=False)
    best=g.iloc[0].to_dict(); top=pd.DataFrame(rows).sort_values('multiple',ascending=False).head(25); top.to_csv(OUT/'top_windows.csv',index=False)
    summary={'method':'Turbo v2 cross-asset momentum rotation + winner concentration + profitable-only pyramiding + DD brake; 10 crypto; rolling 2y every 180d; fee 0.05%','candidate_count':len(g),'best':best,'highest_window':top.iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
