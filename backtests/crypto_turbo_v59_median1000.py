import json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_turbo_v52_dualsleeve as v52

OUT=Path("turbo_v59_output"); OUT.mkdir(exist_ok=True)
def main():
    idx,A=v52.build()
    starts=pd.date_range(max(idx.min(),pd.Timestamp("2018-01-01",tz="UTC")),idx.max()-pd.Timedelta(days=730),freq="180D")
    cfgs=[]
    for alloc in [.60,.65,.70,.75]:
      for breadth in [.80,.85,.90]:
       for m30 in [.85,.90,.95]:
        for lev in [3.0,3.25,3.5,3.75]:
         for hold in [6,8]:
          cfgs.append(dict(alloc=alloc,breadth=breadth,m7=.10,m30=m30,near=.975,accel=.010,lev=lev,lev2=lev*1.30,gain2=.40,cap=lev*1.30,hold=hold))
    cfgs.append(dict(alloc=.70,breadth=.85,m7=.10,m30=.90,near=.975,accel=.010,lev=3.5,lev2=4.55,gain2=.40,cap=4.55,hold=8))
    rows=[]; grids=[]
    core_cache={}
    for st in starts:
      i0=idx.searchsorted(st);i1=min(idx.searchsorted(st+pd.Timedelta(days=730)),len(idx)-1)
      core_cache[str(st)]=v52.core_run(idx,A,i0,i1)[0]
    for ci,cfg in enumerate(cfgs):
      vals=[]
      for st in starts:
        en=st+pd.Timedelta(days=730);i0=idx.searchsorted(st);i1=min(idx.searchsorted(en),len(idx)-1)
        cc=core_cache[str(st)]
        tc,tmdd,tliq,th=v52.tail_run(idx,A,cfg,i0,i1)
        n=min(len(cc),len(tc));combo=(1-cfg["alloc"])*cc[:n]+cfg["alloc"]*tc[:n]
        eq=float(combo[-1]) if n else 1.
        dd=float(np.min(combo/np.maximum.accumulate(combo)-1)*100) if n else 0.
        r=dict(return_pct=(eq-1)*100,multiple=eq,mdd_pct=dd,liquidated=int(eq<=0),tail_liquidated=tliq,tail_hours=th)
        vals.append(r);rows.append({**cfg,"start":st,"end":en,**r})
      d=pd.DataFrame(vals)
      g={**cfg,"windows":len(d),"median_return_pct":d.return_pct.median(),"best_return_pct":d.return_pct.max(),"worst_return_pct":d.return_pct.min(),"median_mdd_pct":d.mdd_pct.median(),"worst_mdd_pct":d.mdd_pct.min(),"profitable_rate_pct":(d.return_pct>0).mean()*100,"liquidations":int(d.liquidated.sum()),"tail_liquidations":int(d.tail_liquidated.sum()),"windows_3000x_plus":int((d.multiple>=3000).sum())}
      g["median1000_target"]=int(g["median_return_pct"]>=1000 and g["liquidations"]==0 and g["tail_liquidations"]==0)
      g["joint_target"]=int(g["median1000_target"] and g["best_return_pct"]>=299900)
      grids.append(g)
      if ci%18==0:print("progress",ci+1,"/",len(cfgs),flush=True)
    gd=pd.DataFrame(grids).sort_values(["median1000_target","median_mdd_pct","worst_mdd_pct","median_return_pct"],ascending=[False,False,False,False])
    rd=pd.DataFrame(rows)
    gd.to_csv(OUT/"grid.csv",index=False);rd.to_csv(OUT/"rolling_2y.csv",index=False);rd.sort_values("multiple",ascending=False).head(100).to_csv(OUT/"top_windows.csv",index=False)
    safe=gd[(gd.median_return_pct>=1000)&(gd.liquidations==0)&(gd.tail_liquidations==0)]
    summary={"method":"V5.9 minimize median MDD subject to median >=1000% and zero tail/portfolio depletion; in-sample only","candidate_count":len(gd),"validation":"in-sample overlapping rolling windows; no slippage/funding or exchange liquidation model","median1000_count":len(safe),"zero_liquidation_configs":int(((gd.liquidations==0)&(gd.tail_liquidations==0)).sum()),"zero_total_liquidation_configs":int((gd.liquidations==0).sum()),"best_return_any":gd.sort_values("median_return_pct",ascending=False).iloc[0].to_dict(),"best_zero_tail":gd[(gd.liquidations==0)&(gd.tail_liquidations==0)].sort_values("median_return_pct",ascending=False).iloc[0].to_dict() if ((gd.liquidations==0)&(gd.tail_liquidations==0)).any() else None,"joint_target_count":int(gd.joint_target.sum()),"best_overall":gd.iloc[0].to_dict(),"best_median1000":safe.sort_values(["median_mdd_pct","worst_mdd_pct"],ascending=False).iloc[0].to_dict() if len(safe) else None,"best_mdd_at_target":safe.sort_values(["median_mdd_pct","worst_mdd_pct"],ascending=False).iloc[0].to_dict() if len(safe) else None,"mdd_below_70_count":int((safe.median_mdd_pct>=-70).sum()),"highest_window":rd.sort_values("multiple",ascending=False).iloc[0].to_dict()}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,default=str),flush=True)
if __name__=="__main__":main()
