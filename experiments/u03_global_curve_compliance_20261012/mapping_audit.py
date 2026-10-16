"""Read-only phase decomposition; nominal geometry only, no physical stepping."""
from physics import *
from curve_reference import GlobalCurveReference
from curve_projector import CurveProjector
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def phase_fit(q,t):
    n=np.arange(8,20);base=-.183889173151851*n+(n%2)*np.pi/2;basis=np.column_stack([np.sin(base),np.cos(base)])
    co=np.linalg.lstsq(basis,np.asarray(q)[:,8:20].T,rcond=None)[0].T;phase=np.unwrap(np.arctan2(co[:,1],co[:,0]));amplitude=np.linalg.norm(co,axis=1);fit=co@basis.T
    return dict(rate_deg_s=float(np.rad2deg(np.polyfit(t,phase,1)[0])),mean_amplitude_rad=float(amplitude.mean()),rms_deg=float(np.rad2deg(np.sqrt(np.mean((fit-np.asarray(q)[:,8:20])**2)))),phase_rad=phase.tolist())

m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(INITIAL);d.qpos[:]=initial['qpos'];d.qvel[:]=initial['qvel'];d.ctrl[:]=initial['ctrl'];mujoco.mj_forward(m,d);p=nodes(d);out={};fig,axs=plt.subplots(1,2,figsize=(12,5))
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A02',BASE/'A02'),('A03',BASE/'A03')]:
    z=np.load(path/'trace.npz');mask=(z['t']>=.6)&(z['t']<=20);fit=phase_fit(z['raw_command'][mask],z['t'][mask]);out[name]=fit;axs[0].plot(z['t'][mask],np.rad2deg(fit['phase_rad']),label=name)
for label,roll,comp,moving in [('no_explicit_roll',0,True,False),('full_fixed',1,True,False),('without_torsion_comp',1,False,False),('moving_family',1,True,True)]:
    ref=GlobalCurveReference(p,d.qpos[7:],roll_sign=roll,moving_family=moving);q0,info=ref.command(0);proj=CurveProjector(m,d.qpos.copy(),info['reference_frame']);proj.command(info['reference_points'],info['reference_frame'])
    if not comp:ref.torsion_compensation=0
    raw=[];projected=[];tt=[]
    for sec in np.arange(.6,20.001,.1):
        S=ref.feed_per_cycle*.02*(sec-.5);q,info=ref.command(S);qp,_=proj.command(info['reference_points'],info['reference_frame']);raw.append(q);projected.append(qp);tt.append(sec)
    curve_fit=phase_fit(raw,tt);projected_fit=phase_fit(projected,tt);out[label]=dict(curvature_projection=curve_fit,exact_hinge_projection=projected_fit,high_level_feedback=False,physical_steps=0,roll_sign=roll,torsion_compensation=comp,moving_family=moving)
    axs[1].plot(tt,np.rad2deg(projected_fit['phase_rad']),label=label);print(label,'field',curve_fit['rate_deg_s'],'projected',projected_fit['rate_deg_s'],flush=True)
axs[0].set(xlabel='t(s)',ylabel='fitted source joint phase(deg)',title='Saved raw commands, not actual material roll');axs[1].set(xlabel='nominal t(s)',ylabel='fitted source joint phase(deg)',title='Geometry-only mapping decomposition');axs[0].legend();axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(BASE/'mapping_audit.png',dpi=140)
(BASE/'mapping_audit.json').write_text(json.dumps(out,indent=2));print('saved mapping_audit.json',flush=True)
