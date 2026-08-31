import json
from pathlib import Path
import numpy as np, pandas as pd
import crypto_universal_10y as base
OUT=Path('turbo_v51_output'); OUT.mkdir(exist_ok=True)
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

BASE={'single':1.,'pair_total':1.2,'p1':1.5,'p2':2.75,'edge':.03,'max_total':3.1,'vol_target':1.28,'vol_floor':.45,'vol_cap':1.55,'dd1':.11,'dd2':.21,'brake1':.70,'brake2':.35,'trail_trigger':.38,'trail':.17,'breadth_thr':.30,'accel_thr':.02,'score_thr':-.01,'hold_hours':24,'single_strong':1.2,'pair_strong':1.4,'vt_strong':1.48,'cap_strong':2.,'p2_strong':3.1,'p3_strong':4.,'p4_strong':5.,'max_strong':4.2}

def run(idx,A,cfg,i0,i1):
 eq=peak=1.; maxdd=0.; w=np.zeros(len(SYMS)); entry=np.full(len(SYMS),np.nan); tpeak=np.full(len(SYMS),np.nan); strong_hold=0; tail_hold=0; tail_hours=0
 for i in range(i0+1,i1+1):
  p=i-1; c0=A['close'][p]; c1=A['close'][i]; score=A['score'][p]; accel=A['accel'][p]; bull=A['bull'][p]; vol=A['vol'][p]; m7=A['m7'][p]; m30=A['m30'][p]; near=A['near60'][p]
  valid=(bull>0)&np.isfinite(score)&np.isfinite(accel)&np.isfinite(c0)&np.isfinite(c1)&(c0>0)&np.isfinite(vol)&(vol>0); inds=np.where(valid)[0]; desired=np.zeros(len(SYMS)); breadth=float(np.nansum(bull>0)/max(1,np.sum(np.isfinite(bull))))
  strong=False; leader=None
  if len(inds):
   ords=inds[np.argsort(score[inds])[::-1]]; top=ords[:2]; leader=top[0]; trig=(breadth>=BASE['breadth_thr'] and accel[leader]>=BASE['accel_thr'] and score[leader]>=BASE['score_thr'])
   if trig:strong_hold=BASE['hold_hours']
   elif strong_hold>0:strong_hold-=1
   strong=strong_hold>0; conc=len(top)==1 or (len(top)>=2 and score[top[0]]-score[top[1]]>=BASE['edge'] and accel[top[0]]>0); picks=[top[0]] if conc else list(top)
   bt=(BASE['single_strong'] if strong else BASE['single']) if conc else (BASE['pair_strong'] if strong else BASE['pair_total']); vt=BASE['vt_strong'] if strong else BASE['vol_target']; cap=BASE['cap_strong'] if strong else BASE['vol_cap']; scales=np.clip(vt/np.array([vol[j] for j in picks]),BASE['vol_floor'],cap); raw=np.ones(len(picks))*(bt/len(picks))*scales
   for q,j in enumerate(picks):
    if np.isfinite(entry[j]) and entry[j]>0:
     gain=c0[j]/entry[j]-1
     if gain>=.08:raw[q]=max(raw[q],BASE['p1']*scales[q])
     if gain>=.22:raw[q]=max(raw[q],(BASE['p2_strong'] if strong else BASE['p2'])*scales[q])
     if strong and gain>=.45:raw[q]=max(raw[q],BASE['p3_strong']*scales[q])
     if strong and gain>=.85:raw[q]=max(raw[q],BASE['p4_strong']*scales[q])
    desired[j]=raw[q]
   tail_trig=(breadth>=cfg['tail_breadth'] and np.isfinite(m7[leader]) and m7[leader]>=cfg['tail_m7'] and np.isfinite(m30[leader]) and m30[leader]>=cfg['tail_m30'] and np.isfinite(near[leader]) and near[leader]>=cfg['tail_near'] and accel[leader]>=cfg['tail_accel'])
   if tail_trig:tail_hold=cfg['tail_hold']
   elif tail_hold>0:tail_hold-=1
   if tail_hold>0:
    tail_hours+=1; ov=cfg['tail_overlay']; desired[leader]+=ov
    if np.isfinite(entry[leader]) and entry[leader]>0:
     gain=c0[leader]/entry[leader]-1
     if gain>=cfg['tail_gain2']: desired[leader]+=cfg['tail_overlay2']
  dd=eq/peak-1
  if dd<=-BASE['dd2']:desired*=BASE['brake2']
  elif dd<=-BASE['dd1']:desired*=BASE['brake1']
  for j in np.where(np.isfinite(entry))[0]:
   if np.isfinite(c0[j]):
    tpeak[j]=max(tpeak[j] if np.isfinite(tpeak[j]) else c0[j],c0[j])
    if tpeak[j]/entry[j]-1>=BASE['trail_trigger'] and c0[j]/tpeak[j]-1<=-BASE['trail']:desired[j]=0
  cap_total=BASE['max_strong'] if strong else BASE['max_total']; cap_total+=cfg['tail_cap_extra'] if tail_hold>0 else 0; tot=desired.sum()
  if tot>cap_total:desired*=cap_total/tot
  eq*=max(0.,1.-np.abs(desired-w).sum()*FEE); ok=(w!=0)&np.isfinite(c0)&np.isfinite(c1)&(c0>0); rets=np.zeros(len(SYMS)); rets[ok]=c1[ok]/c0[ok]-1; eq*=1.+np.dot(w,rets)
  if not np.isfinite(eq) or eq<=0:return {'return_pct':-100.,'multiple':0.,'mdd_pct':-100.,'liquidated':1,'tail_hours':tail_hours}
  exited=(w>0)&(desired<=0); entry[exited]=np.nan; tpeak[exited]=np.nan; entered=(w<=0)&(desired>0); opens=A['open'][i]
  for j in np.where(entered)[0]:
   if np.isfinite(opens[j]) and opens[j]>0:entry[j]=opens[j];tpeak[j]=opens[j]
  w=desired; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1)
 return {'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':maxdd*100,'liquidated':0,'tail_hours':tail_hours}

def main():
 idx,A=build(); start0=max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC')); last=idx.max()-pd.Timedelta(days=730); starts=pd.date_range(start0,last,freq='180D'); cfgs=[]
 for b in [.7,.8,.9]:
  for m30 in [.8,1.2]:
   for near in [.97,.99]:
    for ov in [1.0,1.5,2.0]:
     for ov2 in [1.0,1.5]:
      for hold in [12,24]:cfgs.append({'tail_breadth':b,'tail_m7':.20,'tail_m30':m30,'tail_near':near,'tail_accel':.02,'tail_overlay':ov,'tail_overlay2':ov2,'tail_gain2':.60,'tail_hold':hold,'tail_cap_extra':ov+ov2})
 rows=[]; grids=[]
 for ci,cfg in enumerate(cfgs):
  vals=[]
  for st in starts:
   en=st+pd.Timedelta(days=730); i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1); r=run(idx,A,cfg,i0,i1); vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
  d=pd.DataFrame(vals); g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,'windows_3000x_plus':int((d.multiple>=3000).sum()),'windows_1000x_plus':int((d.multiple>=1000).sum()),'liquidations':int(d.liquidated.sum()),'median_tail_hours':d.tail_hours.median()}
  g['joint_target']=int(g['best_return_pct']>=299900 and g['median_return_pct']>=200 and g['liquidations']==0); g['score']=4000*g['joint_target']+.0015*g['best_return_pct']+2*g['median_return_pct']+250*g['windows_3000x_plus']+100*g['windows_1000x_plus']-.5*abs(g['median_mdd_pct'])-.15*abs(g['worst_mdd_pct']); grids.append(g)
  if ci%12==0:print('progress',ci+1,'/',len(cfgs),flush=True)
 gd=pd.DataFrame(grids).sort_values(['joint_target','windows_3000x_plus','median_return_pct','score'],ascending=False); rd=pd.DataFrame(rows); gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); rd.sort_values('multiple',ascending=False).head(100).to_csv(OUT/'top_windows.csv',index=False)
 joint=gd[(gd.best_return_pct>=299900)&(gd.median_return_pct>=200)&(gd.liquidations==0)]; t3=gd[(gd.best_return_pct>=299900)&(gd.liquidations==0)]; med=gd[(gd.median_return_pct>=200)&(gd.liquidations==0)]; summary={'method':'Turbo v5.1 V4.8 base + independent rare tail overlay; target median>=200%, best>=3000x, zero liquidations','candidate_count':len(gd),'joint_target_count':len(joint),'target_3000_count':len(t3),'median200_count':len(med),'best_overall':gd.iloc[0].to_dict(),'best_joint':joint.iloc[0].to_dict() if len(joint) else None,'best_3000':t3.iloc[0].to_dict() if len(t3) else None,'best_median200':med.sort_values('median_return_pct',ascending=False).iloc[0].to_dict() if len(med) else None,'highest_window':rd.sort_values('multiple',ascending=False).iloc[0].to_dict()}; (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,default=str),flush=True)
if __name__=='__main__':main()
