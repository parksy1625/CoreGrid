import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v3_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(24*7)-1; m30=c/c.shift(24*30)-1
    accel=(m24-m72/3.0) + 0.5*(m72-m7*72/(24*7))
    hh30=df.high.rolling(24*30).max().shift(1)
    hh14=df.high.rolling(24*14).max().shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/24,adjust=False,min_periods=24).mean()/c
    bull=(c>ema)&(ema>ema.shift(72))
    breakout=(c>hh14)&(m24>0)&(m7>0)
    ultra=bull&breakout&(c>ema*1.10)&(atr<0.09)
    score=0.35*m30+0.25*m7+0.15*m72+0.10*m24+0.15*accel
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(int),'ultra':ultra.astype(int),'atr':atr})

def build_panel():
    panels={}
    for s in SYMS:
        print('fetch',s,flush=True)
        d,_=base.fetch_symbol(s); panels[s]=prep(d)
    idx=sorted(set().union(*[set(x.index) for x in panels.values()]))
    idx=pd.DatetimeIndex(idx)
    mats={fld:pd.DataFrame(index=idx,columns=SYMS,dtype=float) for fld in ['open','close','score','accel','bull','ultra','atr']}
    for s,p in panels.items():
        for fld in mats: mats[fld].loc[p.index,s]=p[fld].values
    return idx,mats

def window_test(idx,mats,cfg,start,end):
    mask=(idx>=start)&(idx<=end); dates=idx[mask]
    if len(dates)<1000: return None
    eq=1.0; peak=1.0; maxdd=0.0; positions={}; entry={}; trail_peak={}; liquid=False
    for k in range(1,len(dates)):
        t0,t1=dates[k-1],dates[k]
        close0=mats['close'].loc[t0]; close1=mats['close'].loc[t1]
        scores=mats['score'].loc[t0]; bull=mats['bull'].loc[t0]; ultra=mats['ultra'].loc[t0]; accel=mats['accel'].loc[t0]
        valid=(bull>0)&scores.notna()&close0.notna()&close1.notna()
        ranked=scores[valid].sort_values(ascending=False)
        chosen=list(ranked.index[:2])
        # concentration only if leader has clear edge and positive acceleration
        concentrate=False
        if len(chosen)>=2:
            s1,s2=ranked.iloc[0],ranked.iloc[1]
            concentrate=(s1-s2>=cfg['edge']) and (accel[chosen[0]]>cfg['accel_min'])
        elif len(chosen)==1:
            concentrate=True
        desired={}
        if chosen:
            if concentrate:
                desired[chosen[0]]=cfg['base_single']
            else:
                each=cfg['base_pair']/len(chosen)
                for s in chosen: desired[s]=each
        # anti-martingale on winners only + ultra regime
        for s in list(desired):
            if ultra[s]>0: desired[s]=max(desired[s],cfg['ultra'])
            if s in entry and entry[s]>0:
                gain=close0[s]/entry[s]-1
                if gain>=0.10: desired[s]=max(desired[s],cfg['pyr1'])
                if gain>=0.25: desired[s]=max(desired[s],cfg['pyr2'])
        # equity DD brake
        dd=eq/peak-1
        if dd<=-cfg['dd2']:
            desired={s:w*cfg['brake2'] for s,w in desired.items()}
        elif dd<=-cfg['dd1']:
            desired={s:w*cfg['brake1'] for s,w in desired.items()}
        # trailing profit lock by asset
        for s in list(desired):
            trail_peak[s]=max(trail_peak.get(s,close0[s]),close0[s])
            if s in entry and entry[s]>0 and trail_peak[s]/entry[s]-1>=cfg['lock_arm']:
                if close0[s]/trail_peak[s]-1<=-cfg['trail']:
                    desired.pop(s,None)
        # cap total exposure
        total=sum(desired.values())
        if total>cfg['max_total'] and total>0:
            scale=cfg['max_total']/total; desired={s:w*scale for s,w in desired.items()}
        # fees on turnover
        allsyms=set(positions)|set(desired)
        turn=sum(abs(desired.get(s,0)-positions.get(s,0)) for s in allsyms)
        eq*=max(0,1-turn*FEE)
        # apply returns using old positions over t0->t1
        pret=0.0
        for s,w in positions.items():
            if pd.notna(close0[s]) and pd.notna(close1[s]) and close0[s]>0:
                pret+=w*(close1[s]/close0[s]-1)
        eq*=1+pret
        if eq<=0 or not np.isfinite(eq):
            eq=0; liquid=True; break
        # entries / exits state
        for s in list(entry):
            if desired.get(s,0)<=0:
                entry.pop(s,None); trail_peak.pop(s,None)
        for s,w in desired.items():
            if w>0 and s not in entry:
                op=mats['open'].loc[t1,s]
                if pd.notna(op) and op>0:
                    entry[s]=op; trail_peak[s]=op
        positions=desired
        peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
    return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':int(liquid)}

def main():
    idx,mats=build_panel()
    starts=pd.date_range(max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC')),idx.max()-pd.Timedelta(days=730),freq='180D')
    cfgs=[]
    for base_single,base_pair in [(0.8,1.0),(1.0,1.2)]:
      for ultra in [1.5,2.0]:
       for pyr1,pyr2 in [(1.25,1.75),(1.5,2.25)]:
        for trail in [0.12,0.18]:
         cfgs.append({'base_single':base_single,'base_pair':base_pair,'ultra':ultra,'pyr1':pyr1,'pyr2':pyr2,'trail':trail,
         'edge':0.035,'accel_min':0.0,'dd1':0.10,'dd2':0.20,'brake1':0.6,'brake2':0.25,'lock_arm':0.25,'max_total':2.5})
    rows=[]; grid=[]
    for i,cfg in enumerate(cfgs,1):
        print('cfg',i,'/',len(cfgs),flush=True); vals=[]
        for st in starts:
            r=window_test(idx,mats,cfg,st,st+pd.Timedelta(days=730))
            if r is None: continue
            vals.append(r); rows.append({**cfg,'start':st,'end':st+pd.Timedelta(days=730),**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),
           'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,
           'windows_10x_plus':int((d.multiple>=10).sum()),'windows_30x_plus':int((d.multiple>=30).sum()),'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'liquidations':int(d.liquidated.sum())}
        g['score']=g['median_return_pct']-0.9*abs(g['median_mdd_pct'])-0.4*abs(g['worst_mdd_pct'])+4*g['windows_10x_plus']+12*g['windows_30x_plus']
        grid.append(g)
    gd=pd.DataFrame(grid).sort_values(['windows_500x_plus','windows_100x_plus','windows_30x_plus','windows_10x_plus','score'],ascending=False)
    rd=pd.DataFrame(rows); gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False)
    top=rd.sort_values('multiple',ascending=False).head(30); top.to_csv(OUT/'top_windows.csv',index=False)
    # best under MDD constraints
    under50=gd[(gd.median_mdd_pct>=-50)&(gd.worst_mdd_pct>=-65)]
    best_safe=under50.iloc[0].to_dict() if len(under50) else None
    summary={'method':'Turbo v3 acceleration ranking + top2 diversification / leader concentration + winner-only anti-martingale + trailing profit lock + two-stage DD brake; 10 crypto; rolling 2y every 180d; fee 0.05%',
             'candidate_count':len(gd),'best_overall':gd.iloc[0].to_dict(),'best_mdd_constrained':best_safe,'highest_window':top.iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,default=str),flush=True)

if __name__=='__main__': main()
