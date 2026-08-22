import json
from pathlib import Path
import numpy as np, pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v2_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    mom24=c/c.shift(24)-1; mom7=c/c.shift(24*7)-1
    hh30=df.high.rolling(24*30).max().shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/24,adjust=False,min_periods=24).mean()/c
    score=(c/c.shift(24*30)-1)*0.5+(c/c.shift(24*7)-1)*0.3+(c/c.shift(24)-1)*0.2
    bull=(c>ema)&(ema>ema.shift(72))
    ultra=bull&(c>ema*1.12)&(c>hh30)&(mom24>0)&(mom7>0)&(atr<0.08)
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'bull':bull.astype(float),'ultra':ultra.astype(float)})

def run(M,cfg,start,end):
    idx=M['close'].index; sel=(idx>=start)&(idx<=end); pos=np.where(sel)[0]
    if len(pos)<2:return {'return_pct':0,'multiple':1,'mdd_pct':0,'liquidated':0}
    eq=1.0; peak=1.0; maxdd=0.0; cur=-1; entry=0.0; exp=0.0; liquid=0
    C=M['close'].to_numpy(); O=M['open'].to_numpy(); S=M['score'].to_numpy(); B=M['bull'].to_numpy(); U=M['ultra'].to_numpy()
    for z in range(1,len(pos)):
        i0,i1=pos[z-1],pos[z]
        scores=S[i0].copy(); scores[(B[i0]<=0)|(~np.isfinite(scores))]=-np.inf
        chosen=int(np.argmax(scores)) if np.isfinite(scores).any() else -1
        dd=eq/peak-1.0; brake=cfg['brake'] if dd<=-cfg['dd'] else 1.0
        desired=0.0
        if chosen>=0:
            desired=cfg['turbo'] if U[i0,chosen]>0 else 0.5
            if cur==chosen and entry>0 and np.isfinite(C[i0,chosen]):
                gain=C[i0,chosen]/entry-1
                if gain>0.12: desired=max(desired,cfg['pyr1'])
                if gain>0.25: desired=max(desired,cfg['pyr2'])
            desired*=brake
        turn=(exp+desired) if cur!=chosen else abs(desired-exp)
        eq*=max(0.0,1-turn*FEE)
        if cur>=0 and exp>0 and np.isfinite(C[i0,cur]) and np.isfinite(C[i1,cur]):
            eq*=1+exp*(C[i1,cur]/C[i0,cur]-1)
            if eq<=0: eq=0; liquid=1; break
        if chosen!=cur:
            cur=chosen; entry=0.0 if chosen<0 or not np.isfinite(O[i1,chosen]) else O[i1,chosen]
        exp=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
    return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':liquid}

def main():
    parts={k:[] for k in ['open','close','score','bull','ultra']}
    for s in SYMS:
        print('fetch',s,flush=True); d,_=base.fetch_symbol(s); p=prep(d)
        for k in parts: parts[k].append(p[k].rename(s))
    M={k:pd.concat(v,axis=1).sort_index() for k,v in parts.items()}
    master=M['close'].index
    for k in M: M[k]=M[k].reindex(master)
    cfgs=[]
    for turbo in [1.5,2.0,2.5,3.0]:
      for p1,p2 in [(1.0,1.5),(1.25,2.0)]:
       for dd,br in [(0.10,0.5),(0.15,0.35)]: cfgs.append({'turbo':turbo,'pyr1':p1,'pyr2':p2,'dd':dd,'brake':br})
    starts=pd.date_range(max(master.min(),pd.Timestamp('2018-01-01',tz='UTC')),master.max()-pd.Timedelta(days=730),freq='180D')
    detail=[]; grid=[]
    for ci,cfg0 in enumerate(cfgs,1):
        cfg=dict(cfg0); vals=[]; print('cfg',ci,'/',len(cfgs),cfg,flush=True)
        for st in starts:
            en=st+pd.Timedelta(days=730); r=run(M,cfg,st,en); vals.append(r); detail.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals); row={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,'windows_10x_plus':int((d.multiple>=10).sum()),'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'liquidations':int(d.liquidated.sum())}; grid.append(row)
    g=pd.DataFrame(grid); g['score']=g.median_return_pct-0.7*g.worst_mdd_pct.abs()+2*g.windows_10x_plus+20*g.windows_100x_plus
    g=g.sort_values(['windows_500x_plus','windows_100x_plus','windows_10x_plus','score'],ascending=False)
    dd=pd.DataFrame(detail); dd.to_csv(OUT/'rolling_2y.csv',index=False); g.to_csv(OUT/'grid.csv',index=False)
    top=dd.sort_values('multiple',ascending=False).head(25); top.to_csv(OUT/'top_windows.csv',index=False)
    summary={'method':'Turbo v2 cross-asset momentum rotation + winner concentration + profitable-only pyramiding + DD brake; 10 crypto; rolling 2y every 180d; 0.05% fee','candidate_count':len(g),'best':g.iloc[0].to_dict(),'highest_window':top.iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
