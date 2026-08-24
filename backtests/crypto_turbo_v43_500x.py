import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v43_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    r1=c.pct_change()
    rv7=r1.rolling(168).std()*np.sqrt(24*365)
    rv2=r1.rolling(48).std()*np.sqrt(24*365)
    m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(168)-1; m30=c/c.shift(720)-1
    accel=(m24-m72/3.0)+0.5*(m72-m7*(72/168.0))
    hh14=df.high.rolling(336).max().shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/24,adjust=False,min_periods=24).mean()/c
    bull=(c>ema)&(ema>ema.shift(72))
    ultra=bull&(c>hh14)&(c>ema*1.10)&(m24>0)&(m7>0)&(atr<0.09)
    score=0.35*m30+0.25*m7+0.15*m72+0.10*m24+0.15*accel
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(float),'ultra':ultra.astype(float),'rv7':rv7,'rv2':rv2})

def build_arrays():
    ps={}
    for s in SYMS:
        print('fetch',s,flush=True)
        d,_=base.fetch_symbol(s); ps[s]=prep(d)
    idx=ps[SYMS[0]].index
    for s in SYMS[1:]: idx=idx.union(ps[s].index)
    idx=idx.sort_values(); n=len(idx); m=len(SYMS)
    keys=['open','close','score','accel','bull','ultra','rv7','rv2']
    arr={k:np.full((n,m),np.nan,float) for k in keys}
    pos={t:i for i,t in enumerate(idx)}
    for j,s in enumerate(SYMS):
        p=ps[s]; ii=np.array([pos[t] for t in p.index],dtype=int)
        for k in keys: arr[k][ii,j]=p[k].to_numpy(float)
    return idx,arr

def test_window(idx,A,cfg,i0,i1):
    eq=1.0; peak=1.0; maxdd=0.0
    m=len(SYMS); w=np.zeros(m); entry=np.full(m,np.nan); tpeak=np.full(m,np.nan)
    for i in range(i0+1,i1+1):
        p=i-1
        c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]
        bull=A['bull'][p]; ultra=A['ultra'][p]; rv7=A['rv7'][p]; rv2=A['rv2'][p]
        valid=(bull>0)&np.isfinite(score)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)
        inds=np.where(valid)[0]; desired=np.zeros(m)
        if len(inds):
            ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:2]
            conc=len(top)==1 or (len(top)>=2 and (score[top[0]]-score[top[1]]>=cfg['edge']) and accel[top[0]]>0)
            if conc: desired[top[0]]=cfg['single']
            else: desired[top]=cfg['pair_total']/len(top)
            for j in top:
                # volatility targeting: only scale down in ordinary regimes; ultra can relax cap
                if np.isfinite(rv7[j]) and rv7[j]>0:
                    vt=min(cfg['vol_cap_ultra'] if ultra[j]>0 else cfg['vol_cap'], cfg['vol_target']/rv7[j])
                    desired[j]*=min(1.0,vt)
                if ultra[j]>0:
                    desired[j]=max(desired[j],cfg['ultra'])
                if np.isfinite(entry[j]) and entry[j]>0:
                    gain=c0[j]/entry[j]-1
                    if gain>=0.10: desired[j]=max(desired[j],cfg['p1'])
                    if gain>=0.25: desired[j]=max(desired[j],cfg['p2'])
                    if gain>=0.50: desired[j]=max(desired[j],cfg['p3'])
                    if gain>=1.00 and ultra[j]>0: desired[j]=max(desired[j],cfg['p4'])
                # volatility shock brake
                if np.isfinite(rv2[j]) and np.isfinite(rv7[j]) and rv7[j]>0 and rv2[j]/rv7[j]>=cfg['shock_ratio']:
                    desired[j]*=cfg['shock_mult']
        dd=eq/peak-1
        if dd<=cfg['dd2']: desired*=cfg['dd_mult2']
        elif dd<=cfg['dd1']: desired*=cfg['dd_mult1']
        # trailing only after peak profit milestone was actually reached
        for j in np.where(np.isfinite(entry))[0]:
            if np.isfinite(c0[j]):
                tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
                peak_gain=tpeak[j]/entry[j]-1
                trail=cfg['trail1']
                if peak_gain>=1.0: trail=cfg['trail3']
                elif peak_gain>=0.40: trail=cfg['trail2']
                if peak_gain>=0.25 and c0[j]/tpeak[j]-1<=-trail: desired[j]=0.0
        tot=desired.sum()
        if tot>cfg['max_total']: desired*=cfg['max_total']/tot
        eq*=max(0.0,1.0-np.abs(desired-w).sum()*FEE)
        rets=np.zeros(m); ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets[ok]=c1[ok]/c0[ok]-1
        eq*=1.0+np.dot(w,rets)
        if not np.isfinite(eq) or eq<=0: return {'return_pct':-100.0,'multiple':0.0,'mdd_pct':-100.0,'liquidated':1}
        exited=(w>0)&(desired<=0); entry[exited]=np.nan; tpeak[exited]=np.nan
        entered=(w<=0)&(desired>0); opens=A['open'][i]
        for j in np.where(entered)[0]:
            if np.isfinite(opens[j]) and opens[j]>0: entry[j]=opens[j]; tpeak[j]=opens[j]
        w=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
    return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':0}

def main():
    idx,A=build_arrays(); start0=max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC')); last=idx.max()-pd.Timedelta(days=730)
    starts=pd.date_range(start0,last,freq='180D')
    cfgs=[]
    # focused around V4.2 sweet spot, only 16 combinations
    for vt in [1.0,1.1,1.2,1.3]:
      for p4 in [2.5,3.0]:
       for ultra in [2.0,2.25]:
        cfgs.append({'single':1.0,'pair_total':1.2,'ultra':ultra,'p1':1.5,'p2':2.25,'p3':2.5,'p4':p4,
                     'edge':0.035,'max_total':3.0,'vol_target':vt,'vol_cap':1.25,'vol_cap_ultra':1.75,
                     'shock_ratio':1.8,'shock_mult':0.65,'dd1':-0.10,'dd2':-0.18,'dd_mult1':0.70,'dd_mult2':0.35,
                     'trail1':0.14,'trail2':0.12,'trail3':0.10})
    rows=[]; grids=[]
    for ci,cfg in enumerate(cfgs,1):
        print('cfg',ci,'/',len(cfgs),cfg,flush=True); vals=[]
        for st in starts:
            en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1)
            r=test_window(idx,A,cfg,i0,i1); vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),
           'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,
           'windows_10x_plus':int((d.multiple>=10).sum()),'windows_30x_plus':int((d.multiple>=30).sum()),'windows_100x_plus':int((d.multiple>=100).sum()),
           'windows_200x_plus':int((d.multiple>=200).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'liquidations':int(d.liquidated.sum())}
        g['score']=g['median_return_pct']-0.8*abs(g['median_mdd_pct'])-0.3*abs(g['worst_mdd_pct'])+5*g['windows_10x_plus']+15*g['windows_100x_plus']+30*g['windows_200x_plus']+100*g['windows_500x_plus']
        grids.append(g)
    gd=pd.DataFrame(grids).sort_values(['windows_500x_plus','windows_200x_plus','windows_100x_plus','score'],ascending=False)
    rd=pd.DataFrame(rows); top=rd.sort_values('multiple',ascending=False).head(40)
    gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); top.to_csv(OUT/'top_windows.csv',index=False)
    safe=gd[(gd.median_mdd_pct>=-50)&(gd.worst_mdd_pct>=-65)]
    summary={'method':'Turbo v4.3: V4.2 vol-target base + ultra vol-cap release + winner-only +100% pyramid; rolling 2y/180d; fee 0.05%',
             'candidate_count':len(gd),'best_overall':gd.iloc[0].to_dict(),'best_mdd_constrained':safe.iloc[0].to_dict() if len(safe) else None,
             'highest_window':top.iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
