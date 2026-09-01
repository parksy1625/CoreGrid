import json
from pathlib import Path
import numpy as np, pandas as pd
import crypto_universal_10y as base
OUT=Path('turbo_v52_output'); OUT.mkdir(exist_ok=True)
SYMS=base.SYMBOLS; FEE=base.FEE

def prep(df):
 c=df.close; ema=c.ewm(span=300,adjust=False,min_periods=300).mean(); m24=c/c.shift(24)-1; m72=c/c.shift(72)-1; m7=c/c.shift(168)-1; m30=c/c.shift(720)-1
 accel=(m24-m72/3)+.5*(m72-m7*(72/168)); vol=c.pct_change().rolling(168).std()*np.sqrt(24*365); bull=(c>ema)&(ema>ema.shift(48)); score=.35*m30+.25*m7+.15*m72+.10*m24+.15*accel
 hi60=c.rolling(1440,min_periods=720).max(); near60=c/hi60
 return pd.DataFrame({'open':df.open,'close':c,'score':score,'accel':accel,'bull':bull.astype(float),'vol':vol,'m7':m7,'m30':m30,'near60':near60})

def build():
 ps={s:prep(base.fetch_symbol(s)[0]) for s in SYMS}; idx=ps[SYMS[0]].index
 for s in SYMS[1:]: idx=idx.union(ps[s].index)
 idx=idx.sort_values(); n=len(idx); m=len(SYMS); keys=['open','close','score','accel','bull','vol','m7','m30','near60']; A={k:np.full((n,m),np.nan) for k in keys}; pos={t:i for i,t in enumerate(idx)}
 for j,s in enumerate(SYMS):
  p=ps[s]; ii=np.array([pos[t] for t in p.index])
  for k in keys:A[k][ii,j]=p[k].to_numpy(float)
 return idx,A

def core_run(idx,A,i0,i1):
 eq=peak=1.; maxdd=0.; w=np.zeros(len(SYMS)); entry=np.full(len(SYMS),np.nan); tpeak=np.full(len(SYMS),np.nan); strong_hold=0
 curve=[]
 for i in range(i0+1,i1+1):
  p=i-1; c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; vol=A['vol'][p]
  valid=(bull>0)&np.isfinite(score)&np.isfinite(accel)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)&np.isfinite(vol)&(vol>0); inds=np.where(valid)[0]; desired=np.zeros(len(SYMS)); breadth=float(np.nansum(bull>0)/max(1,np.sum(np.isfinite(bull))))
  strong=False
  if len(inds):
   ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:2]; leader=top[0]
   trig=(breadth>=.30 and accel[leader]>=.02 and score[leader]>=-.01)
   if trig:strong_hold=24
   elif strong_hold>0:strong_hold-=1
   strong=strong_hold>0; conc=len(top)==1 or (len(top)>=2 and score[top[0]]-score[top[1]]>=.03 and accel[top[0]]>0); picks=[top[0]] if conc else list(top)
   bt=(1.2 if strong else 1.0) if conc else (1.4 if strong else 1.2); vt=1.48 if strong else 1.28; cap=2.0 if strong else 1.55; scales=np.clip(vt/np.array([vol[j] for j in picks]),.45,cap); raw=np.ones(len(picks))*(bt/len(picks))*scales
   for q,j in enumerate(picks):
    if np.isfinite(entry[j]) and entry[j]>0:
     gain=c0[j]/entry[j]-1
     if gain>=.08:raw[q]=max(raw[q],1.5*scales[q])
     if gain>=.22:raw[q]=max(raw[q],(3.1 if strong else 2.75)*scales[q])
     if strong and gain>=.45:raw[q]=max(raw[q],4.0*scales[q])
     if strong and gain>=.85:raw[q]=max(raw[q],5.0*scales[q])
    desired[j]=raw[q]
  dd=eq/peak-1
  if dd<=-.21:desired*=.35
  elif dd<=-.11:desired*=.70
  for j in np.where(np.isfinite(entry))[0]:
   if np.isfinite(c0[j]):
    tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
    if tpeak[j]/entry[j]-1>=.38 and c0[j]/tpeak[j]-1<=-.17:desired[j]=0
  cap_total=4.2 if strong else 3.1; tot=desired.sum()
  if tot>cap_total:desired*=cap_total/tot
  eq*=max(0.,1.-np.abs(desired-w).sum()*FEE); ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets=np.zeros(len(SYMS)); rets[ok]=c1[ok]/c0[ok]-1; eq*=1.+np.dot(w,rets)
  if not np.isfinite(eq) or eq<=0: eq=0.
  exited=(w>0)&(desired<=0); entry[exited]=np.nan; tpeak[exited]=np.nan; entered=(w<=0)&(desired>0); opens=A['open'][i]
  for j in np.where(entered)[0]:
   if np.isfinite(opens[j]) and opens[j]>0:entry[j]=opens[j];tpeak[j]=opens[j]
  w=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1 if peak>0 else -1); curve.append(eq)
 return np.array(curve),maxdd*100

def tail_run(idx,A,cfg,i0,i1):
 eq=peak=1.; maxdd=0.; w=np.zeros(len(SYMS)); entry=np.full(len(SYMS),np.nan); hold=0; active=0; curve=[]
 for i in range(i0+1,i1+1):
  p=i-1; c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; m7=A['m7'][p]; m30=A['m30'][p]; near=A['near60'][p]
  valid=(bull>0)&np.isfinite(score)&np.isfinite(accel)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); inds=np.where(valid)[0]; desired=np.zeros(len(SYMS)); breadth=float(np.nansum(bull>0)/max(1,np.sum(np.isfinite(bull))))
  if len(inds):
   leader=inds[np.argmax(score[inds])]
   trig=(breadth>=cfg['breadth'] and np.isfinite(m7[leader]) and m7[leader]>=cfg['m7'] and np.isfinite(m30[leader]) and m30[leader]>=cfg['m30'] and np.isfinite(near[leader]) and near[leader]>=cfg['near'] and accel[leader]>=cfg['accel'])
   if trig:hold=cfg['hold']
   elif hold>0:hold-=1
   if hold>0:
    active+=1; desired[leader]=cfg['lev']
    if np.isfinite(entry[leader]) and entry[leader]>0 and c0[leader]/entry[leader]-1>=cfg['gain2']: desired[leader]=cfg['lev2']
  tot=desired.sum()
  if tot>cfg['cap']:desired*=cfg['cap']/tot
  eq*=max(0.,1.-np.abs(desired-w).sum()*FEE); ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets=np.zeros(len(SYMS)); rets[ok]=c1[ok]/c0[ok]-1; eq*=1.+np.dot(w,rets)
  if not np.isfinite(eq) or eq<=0: eq=0.; curve.extend([0.]*(i1-i+1)); return np.array(curve),-100.,1,active
  exited=(w>0)&(desired<=0); entry[exited]=np.nan; entered=(w<=0)&(desired>0); opens=A['open'][i]
  for j in np.where(entered)[0]:
   if np.isfinite(opens[j]) and opens[j]>0:entry[j]=opens[j]
  w=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1); curve.append(eq)
 return np.array(curve),maxdd*100,0,active

def main():
 idx,A=build(); start0=max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC')); last=idx.max()-pd.Timedelta(days=730); starts=pd.date_range(start0,last,freq='180D'); cfgs=[]
 for alloc in [.05,.10,.15]:
  for b in [.7,.8,.9]:
   for m30 in [.8,1.2]:
    for lev in [3.,5.,7.]:
     for hold in [6,12]: cfgs.append({'alloc':alloc,'breadth':b,'m7':.20,'m30':m30,'near':.99,'accel':.02,'lev':lev,'lev2':lev*1.5,'gain2':.60,'cap':lev*1.5,'hold':hold})
 rows=[]; grids=[]
 for ci,cfg in enumerate(cfgs):
  vals=[]
  for st in starts:
   en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1); cc,cmdd=core_run(idx,A,i0,i1); tc,tmdd,tliq,th=tail_run(idx,A,cfg,i0,i1); n=min(len(cc),len(tc)); cc=cc[:n]; tc=tc[:n]; a=cfg['alloc']; combo=(1-a)*cc+a*tc; eq=float(combo[-1]) if n else 1.; peak=np.maximum.accumulate(combo) if n else np.array([1.]); dd=np.min(combo/peak-1)*100 if n else 0.; liq=int(eq<=0)
   r={'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':dd,'liquidated':liq,'tail_liquidated':tliq,'tail_hours':th,'core_mdd_pct':cmdd,'tail_mdd_pct':tmdd}; vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
  d=pd.DataFrame(vals); g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,'windows_1000x_plus':int((d.multiple>=1000).sum()),'windows_3000x_plus':int((d.multiple>=3000).sum()),'liquidations':int(d.liquidated.sum()),'tail_liquidations':int(d.tail_liquidated.sum()),'median_tail_hours':d.tail_hours.median()}
  g['joint_target']=int(g['best_return_pct']>=299900 and g['median_return_pct']>=200 and g['liquidations']==0); g['score']=5000*g['joint_target']+.002*g['best_return_pct']+2*g['median_return_pct']+300*g['windows_3000x_plus']+100*g['windows_1000x_plus']-.4*abs(g['median_mdd_pct'])-.1*abs(g['worst_mdd_pct']); grids.append(g)
  if ci%12==0:print('progress',ci+1,'/',len(cfgs),flush=True)
 gd=pd.DataFrame(grids).sort_values(['joint_target','median_return_pct','windows_3000x_plus','score'],ascending=False); rd=pd.DataFrame(rows); gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); rd.sort_values('multiple',ascending=False).head(100).to_csv(OUT/'top_windows.csv',index=False)
 joint=gd[(gd.best_return_pct>=299900)&(gd.median_return_pct>=200)&(gd.liquidations==0)]; t3=gd[(gd.best_return_pct>=299900)&(gd.liquidations==0)]; med=gd[(gd.median_return_pct>=200)&(gd.liquidations==0)]; summary={'method':'Turbo v5.2 isolated dual-sleeve: V4.8 core + separately-capitalized tail sleeve; target median>=200%, best>=3000x, zero total liquidations','candidate_count':len(gd),'joint_target_count':len(joint),'target_3000_count':len(t3),'median200_count':len(med),'best_overall':gd.iloc[0].to_dict(),'best_joint':joint.iloc[0].to_dict() if len(joint) else None,'best_3000':t3.sort_values('best_return_pct',ascending=False).iloc[0].to_dict() if len(t3) else None,'best_median200':med.sort_values('median_return_pct',ascending=False).iloc[0].to_dict() if len(med) else None,'highest_window':rd.sort_values('multiple',ascending=False).iloc[0].to_dict()}; (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__':main()
