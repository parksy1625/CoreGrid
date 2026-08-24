import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v4_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(168)-1; m30=c/c.shift(720)-1
    accel=(m24-m72/3.0)+0.5*(m72-m7*(72/168.0))
    score=0.35*m30+0.25*m7+0.15*m72+0.10*m24+0.15*accel
    bull=(c>ema)&(ema>ema.shift(72))
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(float),'ema':ema})

def build_arrays():
    ps={}
    for s in SYMS:
        print('fetch',s,flush=True); d,_=base.fetch_symbol(s); ps[s]=prep(d)
    idx=ps[SYMS[0]].index
    for s in SYMS[1:]: idx=idx.union(ps[s].index)
    idx=idx.sort_values(); n=len(idx); m=len(SYMS)
    arr={k:np.full((n,m),np.nan,float) for k in ['open','close','score','accel','bull','ema']}
    pos={t:i for i,t in enumerate(idx)}
    for j,s in enumerate(SYMS):
        p=ps[s]; ii=np.array([pos[t] for t in p.index],dtype=int)
        for k in arr: arr[k][ii,j]=p[k].to_numpy(float)
    return idx,arr

def test_window(idx,A,cfg,i0,i1):
    nasset=len(SYMS); eq=1.0; peak=1.0; maxdd=0.0
    w=np.zeros(nasset); entry=np.full(nasset,np.nan); tpeak=np.full(nasset,np.nan)
    cooldown=np.zeros(nasset,dtype=int)
    for i in range(i0+1,i1+1):
        p=i-1; cooldown=np.maximum(cooldown-1,0)
        c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; ema=A['ema'][p]
        tradable=np.isfinite(c0)&np.isfinite(c1)&(c0>0)&np.isfinite(score)&np.isfinite(ema)
        valid=tradable&(bull>0)&(cooldown==0)
        breadth=(valid.sum()/tradable.sum()) if tradable.sum() else 0.0
        desired=np.zeros(nasset)
        if breadth>=cfg['breadth_min'] and valid.any():
            inds=np.where(valid)[0]; ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:3]
            # early bull = diversify top3; strong breadth + clear leader = concentrate
            conc=False
            if len(top)==1: conc=True
            elif len(top)>=2: conc=(score[top[0]]-score[top[1]]>=cfg['edge']) and (accel[top[0]]>0) and breadth>=cfg['breadth_strong']
            total=cfg['base_total']
            if breadth>=cfg['breadth_strong']: total=cfg['strong_total']
            if conc:
                desired[top[0]]=total
            else:
                ww=np.array([0.50,0.30,0.20])[:len(top)]; ww=ww/ww.sum(); desired[top]=total*ww
            leader=top[0]
            # ultra only when broad market strong and leader far above EMA
            if breadth>=cfg['breadth_ultra'] and c0[leader]>ema[leader]*cfg['ultra_ema'] and accel[leader]>0:
                desired[:]=0.0; desired[leader]=cfg['ultra_total']
            # anti-martingale: only add to winning leader
            if np.isfinite(entry[leader]) and entry[leader]>0:
                gain=c0[leader]/entry[leader]-1
                if gain>=0.15: desired[leader]=max(desired[leader],cfg['p1'])
                if gain>=0.35: desired[leader]=max(desired[leader],cfg['p2'])
                if gain>=0.70: desired[leader]=max(desired[leader],cfg['p3'])
        # portfolio DD brake
        dd=eq/peak-1
        if dd<=-0.20: desired*=0.0
        elif dd<=-0.15: desired*=0.35
        elif dd<=-0.10: desired*=0.65
        # peak-based trailing profit locks
        held=np.where(np.isfinite(entry))[0]
        for j in held:
            if np.isfinite(c0[j]):
                tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
                peak_profit=tpeak[j]/entry[j]-1; retrace=c0[j]/tpeak[j]-1
                exit_now=False
                if peak_profit>=0.80 and retrace<=-cfg['trail3']: exit_now=True
                elif peak_profit>=0.40 and retrace<=-cfg['trail2']: exit_now=True
                elif peak_profit>=0.20 and retrace<=-cfg['trail1']: exit_now=True
                if exit_now:
                    desired[j]=0.0; cooldown[j]=cfg['cooldown']
        tot=desired.sum()
        if tot>cfg['max_total']: desired*=cfg['max_total']/tot
        # turnover fee then bar return
        eq*=max(0.0,1.0-np.abs(desired-w).sum()*FEE)
        rets=np.zeros(nasset); ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets[ok]=c1[ok]/c0[ok]-1
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
    for breadth_min,breadth_strong,breadth_ultra in [(0.4,0.6,0.7),(0.5,0.7,0.8)]:
      for ultra,p1,p2,p3 in [(1.5,1.1,1.5,2.0),(1.8,1.25,1.7,2.2),(2.0,1.4,1.9,2.4)]:
       for trail1,trail2,trail3 in [(0.14,0.11,0.09),(0.18,0.14,0.10)]:
        cfgs.append({'breadth_min':breadth_min,'breadth_strong':breadth_strong,'breadth_ultra':breadth_ultra,'base_total':0.9,'strong_total':1.2,'ultra_total':ultra,'p1':p1,'p2':p2,'p3':p3,'trail1':trail1,'trail2':trail2,'trail3':trail3,'edge':0.035,'ultra_ema':1.10,'cooldown':24,'max_total':2.5})
    rows=[]; grids=[]
    for ci,cfg in enumerate(cfgs,1):
        print('cfg',ci,'/',len(cfgs),cfg,flush=True); vals=[]
        for st in starts:
            en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1)
            r=test_window(idx,A,cfg,i0,i1); vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,'windows_10x_plus':int((d.multiple>=10).sum()),'windows_30x_plus':int((d.multiple>=30).sum()),'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'liquidations':int(d.liquidated.sum())}
        g['score']=g['median_return_pct']-0.8*abs(g['median_mdd_pct'])-0.35*abs(g['worst_mdd_pct'])+4*g['windows_10x_plus']+15*g['windows_30x_plus']+40*g['windows_100x_plus']; grids.append(g)
    gd=pd.DataFrame(grids).sort_values(['windows_500x_plus','windows_100x_plus','windows_30x_plus','windows_10x_plus','score'],ascending=False)
    rd=pd.DataFrame(rows); gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False)
    top=rd.sort_values('multiple',ascending=False).head(30); top.to_csv(OUT/'top_windows.csv',index=False)
    safe=gd[(gd.median_mdd_pct>=-50)&(gd.worst_mdd_pct>=-65)]
    summary={'method':'Turbo v4: market breadth, acceleration ranking, top3-to-leader concentration, 3-stage winner-only anti-martingale, peak-based 3-stage trailing profit lock, 3-stage portfolio DD brake; 10 crypto; rolling 2y every 180d; fee 0.05%','candidate_count':len(gd),'best_overall':gd.iloc[0].to_dict(),'best_mdd_constrained':safe.iloc[0].to_dict() if len(safe) else None,'highest_window':top.iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
