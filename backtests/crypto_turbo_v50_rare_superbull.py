import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_universal_10y as base

OUT=Path('turbo_v50_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
    c=df.close
    ema=c.ewm(span=300,adjust=False,min_periods=300).mean()
    m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(168)-1; m30=c/c.shift(720)-1
    accel=(m24-m72/3.0)+0.5*(m72-m7*(72/168.0))
    vol=c.pct_change().rolling(24*7).std()*np.sqrt(24*365)
    bull=(c>ema)&(ema>ema.shift(48))
    score=0.35*m30+0.25*m7+0.15*m72+0.10*m24+0.15*accel
    hi30=c.rolling(720,min_periods=240).max(); hi60=c.rolling(1440,min_periods=480).max()
    br30=c/hi30; br60=c/hi60
    return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(float),'vol':vol,'m7':m7,'m30':m30,'br30':br30,'br60':br60})

def build_arrays():
    ps={}
    for s in SYMS:
        d,_=base.fetch_symbol(s); ps[s]=prep(d)
    idx=ps[SYMS[0]].index
    for s in SYMS[1:]: idx=idx.union(ps[s].index)
    idx=idx.sort_values(); n=len(idx); m=len(SYMS)
    keys=['open','close','score','accel','bull','vol','m7','m30','br30','br60']
    A={k:np.full((n,m),np.nan,float) for k in keys}; pos={t:i for i,t in enumerate(idx)}
    for j,s in enumerate(SYMS):
        p=ps[s]; ii=np.array([pos[t] for t in p.index],dtype=int)
        for k in keys: A[k][ii,j]=p[k].to_numpy(float)
    return idx,A

def run(idx,A,cfg,i0,i1):
    eq=peak=1.; maxdd=0.; w=np.zeros(len(SYMS)); entry=np.full(len(SYMS),np.nan); tpeak=np.full(len(SYMS),np.nan)
    strong_hold=0; super_hold=0; strong_hours=0; super_hours=0
    for i in range(i0+1,i1+1):
        p=i-1; c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; vol=A['vol'][p]
        m7=A['m7'][p]; m30=A['m30'][p]; br=A['br30'][p] if cfg['breakout_hours']==720 else A['br60'][p]
        valid=(bull>0)&np.isfinite(score)&np.isfinite(accel)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)&np.isfinite(vol)&(vol>0)
        inds=np.where(valid)[0]; desired=np.zeros(len(SYMS)); breadth=float(np.nansum(bull>0)/max(1,np.sum(np.isfinite(bull))))
        strong=False; supermode=False
        if len(inds):
            ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:2]; leader=top[0]
            strong_trigger=(breadth>=0.30 and accel[leader]>=0.02 and score[leader]>=-0.01)
            if strong_trigger: strong_hold=24
            elif strong_hold>0: strong_hold-=1
            strong=strong_hold>0
            if strong: strong_hours+=1
            super_trigger=(strong and breadth>=cfg['super_breadth'] and np.isfinite(m7[leader]) and np.isfinite(m30[leader]) and np.isfinite(br[leader]) and m7[leader]>=cfg['super_m7'] and m30[leader]>=cfg['super_m30'] and br[leader]>=0.995 and accel[leader]>0)
            if super_trigger: super_hold=12
            elif super_hold>0: super_hold-=1
            supermode=super_hold>0
            if supermode: super_hours+=1
            conc=len(top)==1 or (len(top)>=2 and score[top[0]]-score[top[1]]>=0.03 and accel[top[0]]>0)
            picks=[top[0]] if conc else list(top)
            base_total=1.2 if (strong and conc) else (1.4 if strong else (1.0 if conc else 1.2))
            vt=1.48 if strong else 1.28; cap=2.0 if strong else 1.55
            scales=np.clip(vt/np.array([vol[j] for j in picks]),0.45,cap); raw=np.ones(len(picks))*(base_total/len(picks))*scales
            for q,j in enumerate(picks):
                if np.isfinite(entry[j]) and entry[j]>0:
                    gain=c0[j]/entry[j]-1
                    if gain>=0.08: raw[q]=max(raw[q],1.5*scales[q])
                    if gain>=0.22: raw[q]=max(raw[q],(3.1 if strong else 2.75)*scales[q])
                    if strong and gain>=0.45: raw[q]=max(raw[q],4.0*scales[q])
                    if strong and gain>=0.85: raw[q]=max(raw[q],5.0*scales[q])
                    if supermode and gain>=1.20: raw[q]=max(raw[q],cfg['p5_super']*scales[q])
                desired[j]=raw[q]
        dd=eq/peak-1
        if dd<=-0.21: desired*=0.35
        elif dd<=-0.11: desired*=0.70
        for j in np.where(np.isfinite(entry))[0]:
            if np.isfinite(c0[j]):
                tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
                if tpeak[j]/entry[j]-1>=0.38 and c0[j]/tpeak[j]-1<=-0.17: desired[j]=0.
        max_total=cfg['max_super'] if supermode else (4.2 if strong else 3.1)
        tot=desired.sum()
        if tot>max_total: desired*=max_total/tot
        eq*=max(0.,1.-np.abs(desired-w).sum()*FEE)
        ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets=np.zeros(len(SYMS)); rets[ok]=c1[ok]/c0[ok]-1
        eq*=1.+np.dot(w,rets)
        if not np.isfinite(eq) or eq<=0: return {'return_pct':-100.,'multiple':0.,'mdd_pct':-100.,'liquidated':1,'strong_hours':strong_hours,'super_hours':super_hours}
        exited=(w>0)&(desired<=0); entry[exited]=np.nan; tpeak[exited]=np.nan
        entered=(w<=0)&(desired>0); opens=A['open'][i]
        for j in np.where(entered)[0]:
            if np.isfinite(opens[j]) and opens[j]>0: entry[j]=opens[j]; tpeak[j]=opens[j]
        w=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
    return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':0,'strong_hours':strong_hours,'super_hours':super_hours}

def main():
    idx,A=build_arrays(); start0=max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC')); last=idx.max()-pd.Timedelta(days=730); starts=pd.date_range(start0,last,freq='180D')
    cfgs=[]
    for breadth in [0.60,0.70,0.80]:
      for m7thr in [0.10,0.20]:
       for m30thr in [0.25,0.40]:
        for boh in [720,1440]:
         for p5 in [5.5,6.5]:
          for maxs in [4.8,5.6,6.4]:
           cfgs.append({'super_breadth':breadth,'super_m7':m7thr,'super_m30':m30thr,'breakout_hours':boh,'p5_super':p5,'max_super':maxs})
    rows=[]; grids=[]
    for ci,cfg in enumerate(cfgs):
        vals=[]
        for st in starts:
            en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1); r=run(idx,A,cfg,i0,i1)
            vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),
           'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,
           'windows_100x_plus':int((d.multiple>=100).sum()),'windows_500x_plus':int((d.multiple>=500).sum()),'windows_1000x_plus':int((d.multiple>=1000).sum()),'windows_3000x_plus':int((d.multiple>=3000).sum()),
           'liquidations':int(d.liquidated.sum()),'median_strong_hours':d.strong_hours.median(),'median_super_hours':d.super_hours.median()}
        g['joint_target']=int(g['best_return_pct']>=299900 and g['median_return_pct']>=200 and g['liquidations']==0)
        g['score']=4000*g['joint_target']+0.002*g['best_return_pct']+2.0*g['median_return_pct']+220*g['windows_3000x_plus']+80*g['windows_1000x_plus']-0.6*abs(g['median_mdd_pct'])-0.12*abs(g['worst_mdd_pct'])
        grids.append(g)
        if ci%12==0: print(f'progress {ci+1}/{len(cfgs)}',flush=True)
    gd=pd.DataFrame(grids).sort_values(['joint_target','median_return_pct','windows_3000x_plus','score'],ascending=False); rd=pd.DataFrame(rows)
    gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); rd.sort_values('multiple',ascending=False).head(100).to_csv(OUT/'top_windows.csv',index=False)
    joint=gd[(gd.best_return_pct>=299900)&(gd.median_return_pct>=200)&(gd.liquidations==0)]
    t3=gd[(gd.best_return_pct>=299900)&(gd.liquidations==0)]; med=gd[(gd.median_return_pct>=200)&(gd.liquidations==0)]
    summary={'method':'Turbo v5.0 V4.8 anchor + rare super-bull breakout; target best 2y >=3000x and median >=200%; rolling 2y/180d, fee 0.05%',
      'candidate_count':len(gd),'joint_target_count':len(joint),'target_3000_count':len(t3),'median200_count':len(med),'best_overall':gd.iloc[0].to_dict(),
      'best_joint':joint.iloc[0].to_dict() if len(joint) else None,'best_3000':t3.sort_values('best_return_pct',ascending=False).iloc[0].to_dict() if len(t3) else None,
      'best_median200':med.sort_values('median_return_pct',ascending=False).iloc[0].to_dict() if len(med) else None,'highest_window':rd.sort_values('multiple',ascending=False).iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__': main()
