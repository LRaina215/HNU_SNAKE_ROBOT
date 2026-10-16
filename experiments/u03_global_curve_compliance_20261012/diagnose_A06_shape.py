"""Shape/rigid-pose decomposition of saved A06; no simulation steps."""
from physics import *
from scipy.spatial.transform import Rotation
import json

def register(x,y,ids):
    xx=x[ids]; yy=y[ids]; u,_,v=np.linalg.svd((xx-xx.mean(0)).T@(yy-yy.mean(0)))
    r=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T
    return x@r.T+yy.mean(0)-xx.mean(0)@r.T

z=np.load(BASE/'A06/trace.npz');m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);out=[]
for sec in [0,10,20,30,35,40]:
    i=np.argmin(abs(z['t']-sec));g=z['reference_points'][i];actual=z['points'][i]
    row=dict(t=float(z['t'][i]),actual_head_m=actual[0].tolist(),guide_head_m=g[0].tolist())
    for name,q in [('raw',z['raw_command'][i]),('dispatched',z['ctrl'][i]),('actual',z['qpos'][i,7:])]:
        d.qpos[:]=z['qpos'][i];d.qpos[7:]=q;mujoco.mj_fwdPosition(m,d);p=nodes(d)
        aligned=register(p,g,np.arange(10,22));err=np.linalg.norm(aligned-g,axis=1)
        row[name+'_posterior_registered_front_rms_mm']=float(np.sqrt(np.mean(err[:7]**2))*1000)
        row[name+'_posterior_registered_rear_rms_mm']=float(np.sqrt(np.mean(err[10:]**2))*1000)
        row[name+'_all_registered_rms_mm']=float(np.sqrt(np.mean(np.sum((register(p,g,np.arange(22))-g)**2,axis=1)))*1000)
    err=np.linalg.norm(actual-g,axis=1);row['unregistered_actual_rms_mm']=float(np.sqrt(np.mean(err**2))*1000)
    out.append(row)
(BASE/'A06_shape_pose_decomposition.json').write_text(json.dumps(dict(physical_steps=0,registration='diagnostic rigid Kabsch using posterior nodes10..21; never applied to control',samples=out),indent=2));print(json.dumps(out,indent=2))
