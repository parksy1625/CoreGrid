import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v48_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close; ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(168)-1; m30=c/c.shift(720)-1
    accel=(m24-m72/3.0)+0.5*(m72-m7*(72/168.0))
    vol=c.pct_change().rolling(24*7).std()*np.sqrt(24*365)
    bull=(c>ema)&(ema>ema.shift(48))
    score=0.35*m30+0.25*m7+0.15*m72+0.10*m24+0.15*accel
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(float),'vol':vol})

def build_arrays():
    ps={}
    for s in SYMS:
        d,_=base.fetch_symbol(s); ps[s]=prep(d)
    idx=ps[SYMS[0]].index
    for s in SYMS[1:]: idx=idx.union(ps[s].index)
    idx=idx.sort_values(); n=len(idx); m=len(SYMS)
    A={k:np.full((n,m),np.nan,float) for k in ['open','close','score','accel','bull','vol']}; pos={t:i for i,t in enumerate(idx)}
    for j,s in enumerate(SYMS):
        p=ps[s]; ii=np.array([pos[t] for t in p.index],dtype=int)
        for k in A: A[k][ii,j]=p[k].to_numpy(float)
    return idx,A

def run(idx,A,cfg,i0,i1):
    eq=peak=1.; maxdd=0.; w=np.zeros(len(SYMS)); entry=np.full(len(SYMS),np.nan); tpeak=np.full(len(SYMS),np.nan)
    strong_hold=0; strong_hours=0
    for i in range(i0+1,i1+1):
        p=i-1; c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; vol=A['vol'][p]
        valid=(bull>0)&np.isfinite(score)&np.isfinite(accel)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)&np.isfinite(vol)&(vol>0)
        inds=np.where(valid)[0]; desired=np.zeros(len(SYMS)); breadth=float(np.nansum(bull>0)/max(1,np.sum(np.isfinite(bull))))
        strong=False
        if len(inds):
            ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:2]; leader=top[0]
            trigger=(breadth>=cfg['breadth_thr'] and accel[leader]>=cfg['accel_thr'] and score[leader]>=cfg['score_thr'])
            if trigger: strong_hold=cfg['hold_hours']
            elif strong_hold>0: strong_hold-=1
            strong=strong_hold>0
            if strong: strong_hours+=1
            conc=len(top)==1 or (len(top)>=2 and score[top[0]]-score[top[1]]>=cfg['edge'] and accel[top[0]]>0)
            picks=[top[0]] if conc else list(top)
            base_total=(cfg['single_strong'] if strong else cfg['single']) if conc else (cfg['pair_strong'] if strong else cfg['pair_total'])
            vt=cfg['vt_strong'] if strong else cfg['vol_target']; cap=cfg['cap_strong'] if strong else cfg['vol_cap']
            scales=np.clip(vt/np.array([vol[j] for j in picks]),cfg['vol_floor'],cap); raw=np.ones(len(picks))*(base_total/len(picks))*scales
            for q,j in enumerate(picks):
                if np.isfinite(entry[j]) and entry[j]>0:
                    gain=c0[j]/entry[j]-1
                    if gain>=0.08: raw[q]=max(raw[q],cfg['p1']*scales[q])
                    if gain>=0.22: raw[q]=max(raw[q],(cfg['p2_strong'] if strong else cfg['p2'])*scales[q])
                    if strong and gain>=0.45: raw[q]=max(raw[q],cfg['p3_strong']*scales[q])
                    if strong and gain>=0.85: raw[q]=max(raw[q],cfg['p4_strong']*scales[q])
                desired[j]=raw[q]
        dd=eq/peak-1
        if dd<=-cfg['dd2']: desired*=cfg['brake2']
        elif dd<=-cfg['dd1']: desired*=cfg['brake1']
        for j in np.where(np.isfinite(entry))[0]:
            if np.isfinite(c0[j]):
                tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
                if tpeak[j]/entry[j]-1>=cfg['trail_trigger'] and c0[j]/tpeak[j]-1<=-cfg['trail']: desired[j]=0.
        max_total=cfg['max_strong'] if strong else cfg['max_total']; tot=desired.sum()
        if tot>max_total: desired*=max_total/tot
        eq*=max(0.,1.-np.abs(desired-w).sum()*FEE)
        ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets=np.zeros(len(SYMS)); rets[ok]=c1[ok]/c0[ok]-1
        eq*=1.+np.dot(w,rets)
        if not np.isfinite(eq) or eq<=0: return {'return_pct':-100.,'multiple':0.,'mdd_pct':-100.,'liquidated':1,'strong_hours':strong_hours}
        exited=(w>0)&(desired<=0); entry[exited]=np.nan; tpeak[exited]=np.nan
        entered=(w<=0)&(desired>0); opens=A['open'][i]
        for j in np.where(entered)[0]:
            if np.isfinite(opens[j]) and opens[j]>0: entry[j]=opens[j]; tpeak[j]=opens[j]
        w=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
    return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':0,'strong_hours':strong_hours}

def main():
    idx,A=build_arrays(); start0=max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC')); last=idx.max()-pd.Timedelta(days=730); starts=pd.date_range(start0,last,freq='180D')
    cfgs=[]
    for breadth in [0.30,0.40,0.50]:
      for athr in [-0.01,0.02,0.05]:
       for p2 in [2.50,2.75]:
        for p3 in [3.50,4.00]:
         for p4 in [4.50,5.00]:
          for maxs in [4.20,4.60]:
           cfgs.append({'single':1.0,'pair_total':1.2,'p1':1.5,'p2':p2,'edge':0.03,'max_total':3.1,'vol_target':1.28,'vol_floor':0.45,'vol_cap':1.55,
            'dd1':0.11,'dd2':0.21,'brake1':0.70,'brake2':0.35,'trail_trigger':0.38,'trail':0.17,
            'breadth_thr':breadth,'accel_thr':athr,'score_thr':-0.01,'hold_hours':24,'single_strong':1.2,'pair_strong':1.4,
            'vt_strong':1.48,'cap_strong':2.0,'p2_strong':3.1,'p3_strong':p3,'p4_strong':p4,'max_strong':maxs})
    rows=[]; grids=[]
    for ci,cfg in enumerate(cfgs):
        vals=[]
        for st in starts:
            en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1); r=run(idx,A,cfg,i0,i1)
            vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),
           'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,
           'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'windows_1000x_plus':int((d.multiple>=1000).sum()),
           'windows_3000x_plus':int((d.multiple>=3000).sum()),'liquidations':int(d.liquidated.sum()),'median_strong_hours':d.strong_hours.median()}
        g['joint_target']=int(g['best_return_pct']>=299900 and g['median_return_pct']>=150 and g['liquidations']==0)
        g['score']=2500*g['joint_target']+0.0018*g['best_return_pct']+1.7*g['median_return_pct']+180*g['windows_3000x_plus']+70*g['windows_1000x_plus']-0.65*abs(g['median_mdd_pct'])-0.15*abs(g['worst_mdd_pct'])
        grids.append(g)
        if ci%12==0: print(f'progress {ci+1}/{len(cfgs)}',flush=True)
    gd=pd.DataFrame(grids).sort_values(['joint_target','windows_3000x_plus','median_return_pct','score'],ascending=False); rd=pd.DataFrame(rows)
    gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); rd.sort_values('multiple',ascending=False).head(100).to_csv(OUT/'top_windows.csv',index=False)
    joint=gd[(gd.best_return_pct>=299900)&(gd.median_return_pct>=150)&(gd.liquidations==0)]; t3=gd[(gd.best_return_pct>=299900)&(gd.liquidations==0)]; med=gd[(gd.median_return_pct>=150)&(gd.liquidations==0)]
    summary={'method':'Turbo v4.8 fast regime + p3/p4 pyramiding; target best 2y >=3000x and median >=150%; rolling 2y/180d, fee 0.05%',
      'candidate_count':len(gd),'joint_target_count':len(joint),'target_3000_count':len(t3),'median150_count':len(med),'best_overall':gd.iloc[0].to_dict(),
      'best_joint':joint.iloc[0].to_dict() if len(joint) else None,'best_3000':t3.iloc[0].to_dict() if len(t3) else None,
      'best_median150':med.sort_values('median_return_pct',ascending=False).iloc[0].to_dict() if len(med) else None,'highest_window':rd.sort_values('multiple',ascending=False).iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
