import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v46_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(168)-1; m30=c/c.shift(720)-1
    accel=(m24-m72/3.0)+0.5*(m72-m7*(72/168.0))
    rets=c.pct_change(); vol=rets.rolling(24*7).std()*np.sqrt(24*365)
    bull=(c>ema)&(ema>ema.shift(72))
    score=0.35*m30+0.25*m7+0.15*m72+0.10*m24+0.15*accel
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(float),'vol':vol})

def build_arrays():
    ps={}
    for s in SYMS:
        d,_=base.fetch_symbol(s); ps[s]=prep(d)
    idx=ps[SYMS[0]].index
    for s in SYMS[1:]: idx=idx.union(ps[s].index)
    idx=idx.sort_values(); n=len(idx); m=len(SYMS)
    arr={k:np.full((n,m),np.nan,float) for k in ['open','close','score','accel','bull','vol']}
    pos={t:i for i,t in enumerate(idx)}
    for j,s in enumerate(SYMS):
        p=ps[s]; ii=np.array([pos[t] for t in p.index],dtype=int)
        for k in arr: arr[k][ii,j]=p[k].to_numpy(float)
    return idx,arr

def run(idx,A,cfg,i0,i1):
    eq=1.; peak=1.; maxdd=0.; w=np.zeros(len(SYMS)); entry=np.full(len(SYMS),np.nan); tpeak=np.full(len(SYMS),np.nan)
    for i in range(i0+1,i1+1):
        p=i-1; c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; vol=A['vol'][p]
        valid=(bull>0)&np.isfinite(score)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)&np.isfinite(vol)&(vol>0)
        inds=np.where(valid)[0]; desired=np.zeros(len(SYMS))
        if len(inds):
            ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:2]
            conc=len(top)==1 or (len(top)>=2 and score[top[0]]-score[top[1]]>=cfg['edge'] and accel[top[0]]>0)
            picks=[top[0]] if conc else list(top)
            base_total=cfg['single'] if conc else cfg['pair_total']
            vols=np.array([vol[j] for j in picks])
            scales=np.clip(cfg['vol_target']/vols,cfg['vol_floor'],cfg['vol_cap'])
            raw=np.ones(len(picks))*(base_total/len(picks))*scales
            for q,j in enumerate(picks):
                if np.isfinite(entry[j]) and entry[j]>0:
                    gain=c0[j]/entry[j]-1
                    if gain>=0.10: raw[q]=max(raw[q],cfg['p1']*scales[q])
                    if gain>=0.25: raw[q]=max(raw[q],cfg['p2']*scales[q])
                desired[j]=raw[q]
        dd=eq/peak-1
        if dd<=-cfg['dd2']: desired*=cfg['brake2']
        elif dd<=-cfg['dd1']: desired*=cfg['brake1']
        held=np.where(np.isfinite(entry))[0]
        for j in held:
            if np.isfinite(c0[j]):
                tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
                peak_gain=tpeak[j]/entry[j]-1
                if peak_gain>=cfg['trail_trigger'] and c0[j]/tpeak[j]-1<=-cfg['trail']:
                    desired[j]=0.0
        tot=desired.sum()
        if tot>cfg['max_total']: desired*=cfg['max_total']/tot
        eq*=max(0.,1.-np.abs(desired-w).sum()*FEE)
        ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)
        rets=np.zeros(len(SYMS)); rets[ok]=c1[ok]/c0[ok]-1
        eq*=1.+np.dot(w,rets)
        if not np.isfinite(eq) or eq<=0: return {'return_pct':-100.,'multiple':0.,'mdd_pct':-100.,'liquidated':1}
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
    # Focused bridge search around the two V4.5 islands:
    # tail island: vt=1.30/cap=1.70/max=3.0/dd=.13/.24 -> 755x
    # median island: vt=1.20~1.30/cap=1.50/max=3.0/dd=.11/.20 -> median >100%
    ddsets=[(0.11,0.20,0.70,0.35),(0.12,0.22,0.72,0.38),(0.13,0.24,0.75,0.40)]
    for vt in [1.20,1.25,1.30,1.35]:
      for cap in [1.50,1.60,1.70]:
       for p2 in [2.25,2.50,2.75,3.00]:
        for d1,d2,b1,b2 in ddsets:
         cfgs.append({'single':1.0,'pair_total':1.2,'p1':1.5,'p2':p2,'edge':0.035,'max_total':3.20,'vol_target':vt,'vol_floor':0.45,'vol_cap':cap,'dd1':d1,'dd2':d2,'brake1':b1,'brake2':b2,'trail_trigger':0.35,'trail':0.16})
    rows=[]; grids=[]
    for ci,cfg in enumerate(cfgs):
        vals=[]
        for st in starts:
            en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1)
            r=run(idx,A,cfg,i0,i1); vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,'windows_10x_plus':int((d.multiple>=10).sum()),'windows_30x_plus':int((d.multiple>=30).sum()),'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'windows_1000x_plus':int((d.multiple>=1000).sum()),'liquidations':int(d.liquidated.sum())}
        g['joint_target']=int(g['best_return_pct']>=99900 and g['median_return_pct']>=100 and g['liquidations']==0)
        g['score']=1000*g['joint_target']+0.0025*g['best_return_pct']+1.0*g['median_return_pct']-0.6*abs(g['median_mdd_pct'])-0.12*abs(g['worst_mdd_pct'])+100*g['windows_1000x_plus']+30*g['windows_500x_plus']
        grids.append(g)
        if ci%12==0: print(f'progress {ci+1}/{len(cfgs)}',flush=True)
    gd=pd.DataFrame(grids).sort_values(['joint_target','windows_1000x_plus','median_return_pct','score'],ascending=False)
    rd=pd.DataFrame(rows); gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); rd.sort_values('multiple',ascending=False).head(60).to_csv(OUT/'top_windows.csv',index=False)
    joint=gd[(gd.best_return_pct>=99900)&(gd.median_return_pct>=100)&(gd.liquidations==0)]
    med=gd[(gd.median_return_pct>=100)&(gd.liquidations==0)]
    thousand=gd[(gd.best_return_pct>=99900)&(gd.liquidations==0)]
    summary={'method':'Turbo v4.6 focused dual-target bridge search; winner-only p2 expansion; rolling 2y/180d, fee 0.05%','candidate_count':len(gd),'joint_target_count':len(joint),'thousand_target_count':len(thousand),'median100_target_count':len(med),'best_overall':gd.iloc[0].to_dict(),'best_joint':joint.iloc[0].to_dict() if len(joint) else None,'best_1000x':thousand.iloc[0].to_dict() if len(thousand) else None,'best_median100':med.sort_values('median_return_pct',ascending=False).iloc[0].to_dict() if len(med) else None,'highest_window':rd.sort_values('multiple',ascending=False).iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
