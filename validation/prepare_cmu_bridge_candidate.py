"""Bounded offline preparation; never advances any physical MuJoCo data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,time
import numpy as np,mujoco
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.cmu_window_gait import calibrate
from snake_control.cmu_bridge_candidate import BridgeCandidate
from validation.cmu_window_experiment import BASE,IMODEL
from validation.suspended_supply_experiment import write,default

def prepare():
    out=BASE/'C03_PREPARE';out.mkdir(exist_ok=False);m=mujoco.MjModel.from_xml_path(str(IMODEL));A,w,ph,_,cs=calibrate(m);c=BridgeCandidate(m,A,w,ph);results=[];start=time.monotonic()
    bounds=([-np.pi,.1,-.5,-np.pi,.70,-.35,0,0],[np.pi,np.radians(88),.6,np.pi,.96,.35,.75,.75])
    for sign in [-1,1]:
        x0=[sign*np.pi/2,1.0,.05,.4,.82,0,.6,.5];fit=least_squares(c.residual,x0,bounds=bounds,max_nfev=180,ftol=1e-6,xtol=1e-6,gtol=1e-6,diff_step=1e-4);x=fit.x;c.start(x);records=[]
        for s in np.linspace(0,1,101):
            r=c.pose(x,s);cc=r['centers'][:c.front];rr=np.linalg.norm(cc[:,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(cc[:,2]-1,cc[:,0]-1.1));record=dict(s=s,q_rad=r['q'],qpos=r['qpos'],points_mm=r['points']*1000,pipe_depth_mm=float(r['pipe_depth'].max()*1000),self_depth_mm=float(r['self_depth'].max()*1000),target_contact_count=r['target_contact_count'],source_contact_count=r['source_contact_count'],front_target_radius_mm=rr*1000,front_wrap_deg=float(np.degrees(np.ptp(phi))),unsupported_gravity_torque_nm=r['unsupported_gravity_torque_nm']);records.append(record)
        final=records[-1];initial=records[0];maxpd=max(r['pipe_depth_mm'] for r in records);maxsd=max(r['self_depth_mm'] for r in records);rr=np.asarray(final['front_target_radius_mm']);head_mid=np.asarray(records[50]['points_mm'])[0];hook=bool(np.all((rr>75)&(rr<105)) and final['front_wrap_deg']>180 and np.asarray(final['points_mm'])[:8,1].min()>150)
        initialp=np.asarray(initial['points_mm'])/1000;entry=np.asarray(final['points_mm'])[c.front]/1000;gap=np.linalg.norm(entry-initialp[c.bend]);wrap=.75*np.hypot(2*np.pi*.088,.18);margin=c.bend*.0645-gap-wrap
        feasible=bool(initial['target_contact_count']==0 and initial['pipe_depth_mm']<1.5 and maxpd<1.5 and maxsd<.5 and hook and margin>=0 and head_mid[0]<1100 and head_mid[1]>0 and head_mid[2]<1000)
        result=dict(seed_head_phase_sign=sign,parameters=x,global_parameter_names=['head_phase_offset','bridge_amplitude','bridge_spatial','bridge_phase','source_exit_x','source_polar_rotation','head_mid_loosening','bridge_mid_loosening'],A=A,w=w,source_phase=ph,front_target_rods=c.front,bridge_rods=c.bend-c.front,release_rods=c.bend,posterior_source_rods=21-c.bend,release_length_mm=c.bend*.0645*1000,actual_reference_entry_gap_mm=gap*1000,continuous_270deg_wrap_mm=wrap*1000,length_margin_mm=margin*1000,cost=float(fit.cost),nfev=fit.nfev,maximum_reference_pipe_depth_mm=maxpd,maximum_reference_self_depth_mm=maxsd,final_front_target_radius_mm=rr,final_front_wrap_deg=final['front_wrap_deg'],head_mid_mm=head_mid,reference_hook_shape=hook,geometrically_executable=feasible,physical_steps=0,classification='Offline three sine-segment C03 adaptation candidate; not paper Algorithm2, not executed',records=records);results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='records'},default=default),flush=True)
    best=min(results,key=lambda r:(not r['geometrically_executable'],r['cost']));write(out/'candidate_search.json',results);write(out/'candidate.json',best)
    fig=plt.figure(figsize=(17,10))
    for i,j in enumerate([0,20,40,60,80,100]):
        r=best['records'][j];p=np.asarray(r['points_mm'])/1000;ax=fig.add_subplot(2,3,i+1,projection='3d');ax.plot(*p[:c.front+1].T,'o-',c='green',ms=3,label='Target-wave prefix');ax.plot(*p[c.front:c.bend+1].T,'o-',c='orange',ms=3,label='Bridge-wave segment');ax.plot(*p[c.bend:].T,'o-',c='navy',ms=3,label='Retained source');ax.scatter(*p[0],c='red',label='Head');ax.plot([.4,1.1],[0,0],[1,1],lw=10,c='gray',alpha=.3);ax.plot([1.1,1.1],[0,.6],[1,1],lw=10,c='gray',alpha=.3);ax.set(title=f'PRIVATE FK s={r["s"]:.2f}, pipe={r["pipe_depth_mm"]:.2f}mm',xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)');ax.view_init(27,140);ax.legend(fontsize=7)
    fig.suptitle('C03 whole-prefix release and hooking shape / original source M12 frame / NOT actual dynamics');fig.tight_layout();fig.savefig(out/'complete_joint_shape_candidate.png',dpi=150);plt.close(fig);write(out/'preparation_result.json',dict(geometrically_executable=best['geometrically_executable'],physical_steps=0,wall_seconds=time.monotonic()-start,maximum_parameter_seeds=2,selected_candidate_seed=best['seed_head_phase_sign']));print('PREPARATION COMPLETE',time.monotonic()-start,flush=True)

if __name__=='__main__':prepare()
