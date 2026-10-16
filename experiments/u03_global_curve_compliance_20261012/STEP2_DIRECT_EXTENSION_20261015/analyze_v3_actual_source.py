"""Read the preserved V3.2 actual run; no additional dynamics."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
from contact_sensor import known_source
reg=json.loads((C/'REGISTERED.json').read_text());result=json.loads((C/'result.json').read_text());z=np.load(C/'trace.npz');m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('Read-only analysis')))
def bodies(qpos):
    d.qpos[:]=qpos;mujoco.mj_fwdPosition(m,d);return d.xpos[h:h+21].copy(),d.xmat[h:h+21].reshape(21,3,3).copy()
release=int(np.argmin(abs(z['t']-result['first_release_t'])));P0,R0=bodies(z['qpos'][0]);Ps,Rs=bodies(z['qpos'][release]);state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_fwdPosition(m,d);finalq=d.qpos.copy();Pf,Rf=bodies(finalq)
rows=[]
for j in [0,release,int(np.argmin(abs(z['t']-4.))),len(z['t'])-1]:
    p,r=bodies(z['qpos'][j]);start=int(z['source_guard_material_start'][j]);delta=p[start:]-Ps[start:];angle=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',Rs[start:],r[start:])-1)/2,-1,1)))
    phi=np.angle(np.exp(1j*(np.arctan2(p[start:,2]-1,p[start:,1])-np.arctan2(Ps[start:,2]-1,Ps[start:,1]))))
    rows.append(dict(t=float(z['t'][j]),phase=str(z['controller_phase'][j]),alpha=float(z['alpha'][j]),beta=float(z['beta'][j]),S=float(z['S'][j]),guard_start=start,
        maximum_stable_source_motion_mm=float(np.linalg.norm(delta,axis=1).max()*1000),maximum_source_Xmotion_mm=float(abs(delta[:,0]).max()*1000),maximum_orientation_deg=float(angle.max()),maximum_polar_change_deg=float(abs(phi).max()*180/np.pi),
        source_vertical_N=float(z['eval_pipe_world_N'][j,:2,:,2].sum()),receiver_straight_vertical_N=float(z['eval_pipe_world_N'][j,2,:,2].sum()),receiver_outlet_vertical_N=float(z['eval_pipe_world_N'][j,3,:,2].sum()),
        command_filter_factor=float(z['command_filter_factor'][j]),joint_tracking_peak_deg=float(abs(z['qpos'][j,qa]-z['ctrl'][j]).max()*180/np.pi)))
start=14 if str(z['controller_phase'][-1])=='front_wrap' else 8;delta=Pf[start:]-Ps[start:];angle=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',Rs[start:],Rf[start:])-1)/2,-1,1)))
final=dict(t=result['sim_time'],guard_start=start,maximum_source_motion_mm=float(np.linalg.norm(delta,axis=1).max()*1000),maximum_source_Xmotion_mm=float(abs(delta[:,0]).max()*1000),maximum_source_orientation_deg=float(angle.max()),worst_orientation_material=start+int(np.argmax(angle)))
# Map the first controller-unknown contacts against original collision geometry.
evidence=[]
for line in (C/'contact_evidence.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['t']>=result['first_raw_front_contact_t']-1e-8:
        for ct in row['contacts']:
            for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key],float)
        unknown=[ct for ct in row['contacts'] if ct['material']<8 and not known_source(ct,reg['source_known'])]
        evidence.append(dict(t=row['t'],unknown=[dict(material=ct['material'],point=ct['point'].tolist(),normal=ct['normal'].tolist(),source_radial_residual_mm=float((np.linalg.norm(ct['point'][1:]-[0,1])-.0565)*1000),source_miter_residual_mm=float((ct['point'][0]+ct['point'][1]-1.1)*1000)) for ct in unknown]))
        if len(evidence)>=4:break
# Independent fixed-bearing virtual work checks the sign of gravity preload.
j=release;plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),z['qpos'][j],z['ctrl'][j],reg['kinematic_solver']);plan.observe(z['qpos'][j],8);q=z['qpos'][j,qa].copy();implemented=plan.gravity_torque(q,8)[:8]
def potential(v):
    plan.fk(v);return -sum(plan.m.body_mass[plan.head+i]*plan.m.opt.gravity@plan.d.xipos[plan.head+i] for i in range(8))
gradient=[]
for k in range(8):
    plus=q.copy();minus=q.copy();plus[k]+=1e-6;minus[k]-=1e-6;gradient.append((potential(plus)-potential(minus))/2e-6)
gradient=np.asarray(gradient)
gravity=dict(scope='Fixed-source-bearing virtual work only, no contact/load dynamics and no claim of sole failure cause',implemented_torque_Nm=implemented.tolist(),holding_torque_from_potential_gradient_Nm=gradient.tolist(),same_sign_error_Nm=float(abs(implemented-gradient).max()),opposite_sign_error_Nm=float(abs(implemented+gradient).max()))
out=dict(simulation_steps_by_analysis=0,actual_run=result,selected_actual_rows=rows,final_source_guard=final,first_unknown_contact_evidence=evidence,gravity_preload_sign_audit=gravity,
    first_true_receiver_outlet_contact_t=next((float(t) for t,v in zip(z['t'],z['eval_pipe_normal_N'][:,3,:8].sum(1)) if v>.02),None),first_true_receiver_straight_contact_t=next((float(t) for t,v in zip(z['t'],z['eval_pipe_normal_N'][:,2,:8].sum(1)) if v>.02),None))
(C/'SOURCE_DRIFT_AND_GRAVITY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
