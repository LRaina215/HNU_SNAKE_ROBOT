"""Few saved actual states: conservative force-only source wrench feasibility."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.optimize import linprog
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS';C=B/'T_UPDATE';sys.path.insert(0,str(B))
from contact_sensor import known_source
reg=json.loads((C/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;motor=m.jnt_dofadr[m.actuator_trnid[:,0]];records=[json.loads(line) for line in (C/'contact_evidence.jsonl').read_text(encoding='utf-8').splitlines()];rows=[]
for target in [2.1,5.,10.,15.]:
    j=int(np.argmin(abs(z['t']-target)));row=min(records,key=lambda r:abs(r['t']-z['t'][j]));d.qpos[:]=z['qpos'][j];mujoco.mj_fwdPosition(m,d);gravity=np.zeros(m.nv)
    for i in range(21):
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,h+i);gravity-=jp.T@(m.body_mass[h+i]*m.opt.gravity)
    cols=[];ids=[];measured=[];bases=[];contacts=[]
    for ct in row['contacts']:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key],float)
        if ct['material']<8 or not known_source(ct,reg['source_known']):continue
        n=ct['normal'];e=np.array([1.,0.,0.]);e-=n*(n@e)
        if np.linalg.norm(e)<1e-6:e=np.cross(n,[0.,1.,0.])
        e/=np.linalg.norm(e);b=np.cross(n,e);phi=np.linspace(0,2*np.pi,16,endpoint=False);basis=np.column_stack([n,n[:,None]+np.cos(phi)[None,:]*e[:,None]+np.sin(phi)[None,:]*b[:,None]])
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,ct['point'],h+ct['material']);cols.append(jp.T@basis);bases.append(basis);ids.append(ct['material']);measured.append(ct['normal_N']);contacts.append(ct)
    K=np.column_stack(cols);limit=2.30;sol=linprog(np.ones(K.shape[1]),A_ub=np.vstack([K[motor],-K[motor]]),b_ub=np.r_[gravity[motor]+limit,-gravity[motor]+limit],A_eq=K[:6],b_eq=gravity[:6],bounds=(0,None),method='highs')
    item=dict(t=float(z['t'][j]),feasible=bool(sol.success),contact_materials=ids,conservative_friction_mu=1.,contact_torque_ignored=True,actual_retained_normal_N=float(sum(measured)),actual_joint17_torque_Nm=float(z['torque'][j,17]))
    if sol.success:
        weights=sol.x.reshape(len(ids),17);tau=gravity[motor]-K[motor]@sol.x;force=np.array([basis@w for basis,w in zip(bases,weights)]);item.update(allocated_normal_N=float(sol.x.sum()),allocated_contact_normal_N=weights.sum(1).tolist(),allocated_world_force_N=force.tolist(),holding_joint_torque_Nm=tau.tolist(),holding_peak_Nm=float(abs(tau).max()),holding_joint17_Nm=float(tau[17]),base_balance_error=float(abs(K[:6]@sol.x-gravity[:6]).max()))
    rows.append(item)
out=dict(simulation_steps=0,scope='Four saved V3.3 actual poses and unlabeled known-source point/normal records; rearM8-M20 only; force-only friction cones inside physical mu2 cone. No receiver input, no initial change, no rollout.',records=rows,all_feasible=all(r['feasible'] for r in rows),limits='Feasible force allocation does not prove elastic/contact compatibility or realization by position servos. It supports testing an explicit load/preload feedback layer rather than assuming fixed joint targets hold world pose.')
(C/'SOURCE_WRENCH_DIAGNOSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
