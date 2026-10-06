"""Compare connections of the same saved poses; forward geometry only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation,Slerp
from curve_local_path import LocalPathFit,OUT


def evaluate(feeds,states,mode,d):
    index=min(np.searchsorted(feeds,d,side='right')-1,len(feeds)-2)
    index=max(index,0)
    u=(d-feeds[index])/(feeds[index+1]-feeds[index])
    if mode=='coupled_linear':
        state=states[index]+u*(states[index+1]-states[index])
    else:
        state=PchipInterpolator(feeds,states,axis=0)(d)
    state[3:6]=Slerp(feeds,Rotation.from_rotvec(states[:,3:6]))(d).as_rotvec()
    return state


def main():
    fit=LocalPathFit()
    with np.load(OUT/'reference_discrete.npz') as z:
        feeds,states=z['feed_mm'],z['states']
    samples=np.unique(np.r_[np.arange(0,feeds[-1],.25),feeds,(feeds[:-1]+feeds[1:])/2,feeds[-1]])
    results=[]
    for mode in ('mixed_pchip_slerp','coupled_linear'):
        failures=[]
        for feed in samples:
            x=evaluate(feeds,states,mode,feed)
            fit.set(x);mujoco.mj_forward(fit.m,fit.d)
            negative=[dict(geom1=int(c.geom1),geom2=int(c.geom2),distance_mm=float(c.dist*1000))
                      for c in fit.d.contact if c.dist<0]
            if negative:failures.append(dict(feed_mm=float(feed),contacts=negative))
        results.append(dict(mode=mode,physics_steps=0,checked_count=len(samples),
            failure_count=len(failures),maximum_penetration_mm=max([-c['distance_mm'] for r in failures for c in r['contacts']],default=0),failures=failures))
        print(json.dumps({k:v for k,v in results[-1].items() if k!='failures'}),flush=True)
    (OUT/'connection_diagnosis.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
