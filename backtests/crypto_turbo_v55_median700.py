import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_turbo_v52_dualsleeve as v52

OUT=Path('turbo_v55_output'); OUT.mkdir(exist_ok=True)

def main():
    idx,A=v52.build()
    start0=max(idx.min(),pd.Timestamp('2018-01-01',tz='UTC'))
    last=idx.max()-pd.Timedelta(days=730)
    starts=pd.date_range(start0,last,freq='180D')
    cfgs=[]
    # V5.4 best sat at the aggressive edge. Push allocation/exposure further,
    # but retain isolated sleeves so core principal cannot be liquidated by tail.
    for alloc in [.45,.50,.55,.60]:
      for b in [.65,.70,.75]:
       for m30 in [.70,.85,.95]:
        for lev in [4.0,5.0,6.0]:
         for hold in [4,8]:
          cfgs.append({'alloc':alloc,'breadth':b,'m7':.10,'m30':m30,'near':.975,'accel':.010,'lev':lev,'lev2':lev*1.30,'gain2':.40,'cap':lev*1.30,'hold':hold})
    rows=[]; grids=[]
    for ci,cfg in enumerate(cfgs):
        vals=[]
        for st in starts:
            en=st+pd.Timedelta(days=730)
            i0=idx.searchsorted(st); i1=min(idx.searchsorted(en),len(idx)-1)
            cc,cmdd=v52.core_run(idx,A,i0,i1)
            tc,tmdd,tliq,th=v52.tail_run(idx,A,cfg,i0,i1)
            n=min(len(cc),len(tc)); cc=cc[:n]; tc=tc[:n]
            a=cfg['alloc']; combo=(1-a)*cc+a*tc
            eq=float(combo[-1]) if n else 1.
            if n:
                peak=np.maximum.accumulate(combo); dd=float(np.min(combo/peak-1)*100)
            else: dd=0.
            liq=int(eq<=0)
            r={'return_pct':(eq-1)*100,'multiple':eq,'mdd_pct':dd,'liquidated':liq,'tail_liquidated':tliq,'tail_hours':th,'core_mdd_pct':cmdd,'tail_mdd_pct':tmdd}
            vals.append(r); rows.append({**cfg,'start':st,'end':en,**r})
        d=pd.DataFrame(vals)
        g={**cfg,'windows':len(d),'median_return_pct':d.return_pct.median(),'best_return_pct':d.return_pct.max(),'worst_return_pct':d.return_pct.min(),'median_mdd_pct':d.mdd_pct.median(),'worst_mdd_pct':d.mdd_pct.min(),'profitable_rate_pct':(d.return_pct>0).mean()*100,'windows_1000x_plus':int((d.multiple>=1000).sum()),'windows_3000x_plus':int((d.multiple>=3000).sum()),'liquidations':int(d.liquidated.sum()),'tail_liquidations':int(d.tail_liquidated.sum()),'median_tail_hours':d.tail_hours.median()}
        g['median700_target']=int(g['median_return_pct']>=700 and g['liquidations']==0)
        g['joint_target']=int(g['best_return_pct']>=299900 and g['median_return_pct']>=700 and g['liquidations']==0)
        g['score']=9000*g['joint_target']+3200*g['median700_target']+.0012*g['best_return_pct']+4.0*g['median_return_pct']+400*g['windows_3000x_plus']+140*g['windows_1000x_plus']-.55*abs(g['median_mdd_pct'])-.16*abs(g['worst_mdd_pct'])
        grids.append(g)
        if ci%18==0: print('progress',ci+1,'/',len(cfgs),flush=True)
    gd=pd.DataFrame(grids).sort_values(['joint_target','median700_target','median_return_pct','windows_3000x_plus','score'],ascending=False)
    rd=pd.DataFrame(rows)
    gd.to_csv(OUT/'grid.csv',index=False); rd.to_csv(OUT/'rolling_2y.csv',index=False); rd.sort_values('multiple',ascending=False).head(100).to_csv(OUT/'top_windows.csv',index=False)
    joint=gd[(gd.best_return_pct>=299900)&(gd.median_return_pct>=700)&(gd.liquidations==0)]
    t3=gd[(gd.best_return_pct>=299900)&(gd.liquidations==0)]
    med=gd[(gd.median_return_pct>=700)&(gd.liquidations==0)]
    summary={'method':'Turbo v5.5 median-700 push: isolated dual-sleeve aggressive expansion from v5.4; target median>=700%, best>=3000x, zero total liquidations','candidate_count':len(gd),'joint_target_count':len(joint),'target_3000_count':len(t3),'median700_count':len(med),'best_overall':gd.iloc[0].to_dict(),'best_joint':joint.iloc[0].to_dict() if len(joint) else None,'best_3000':t3.sort_values('best_return_pct',ascending=False).iloc[0].to_dict() if len(t3) else None,'best_median700':med.sort_values('median_return_pct',ascending=False).iloc[0].to_dict() if len(med) else None,'highest_window':rd.sort_values('multiple',ascending=False).iloc[0].to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,default=str),flush=True)

if __name__=='__main__': main()
