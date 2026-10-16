"""Continuous native-geometry sweep of V3's archived source-held prefix."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json,time,numpy as np,mujoco
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';O=B/'PRECONTACT_EVALUATION';ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
modelpath=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml';m=mujoco.MjModel.from_xml_path(str(modelpath));m.opt.enableflags|=64;physical=mujoco.MjModel.from_xml_path(str(modelpath));d=mujoco.MjData(m);actual=mujoco.MjData(physical)
z=np.load(O/'PRECONTACT_PREFIX.npz');initial=np.load(B/'GIVEN_INITIAL.npz');q0=initial['qpos'];qa=m.jnt_qposadr[m.actuator_trnid[:,0]];head=m.body('head').id;n=8;anchor=head+n
d.qpos[:]=q0;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);anchorP=d.xpos[anchor].copy();anchorR=d.xmat[anchor].reshape(3,3).copy();rootR=Rotation.from_quat(q0[3:7][[1,2,3,0]]).as_matrix();reference=nodes(d).copy()
geoms=[int(m.body_geomadr[head+j]) for j in range(21)];pipes=[m.geom(name).id for name in ['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04']]
pairs=[(geoms[i],g,i,-1,'source' if k<2 else 'receiver') for i in range(n) for k,g in enumerate(pipes)]+[(geoms[i],geoms[j],i,j,'self') for i in range(n) for j in range(i+2,21)]
cache={};maximum_original_depth=np.zeros(4);maximum_source_motion=0.;minimum_gap=dict(source=np.inf,receiver=np.inf,self=np.inf);start=time.time()
def evaluate(t):
    global maximum_source_motion
    key=round(float(t),13)
    if key in cache:return cache[key]
    q=np.array([np.interp(t,z['t'],z['qpos'][:,j]) for j in qa[:n]]);d.qpos[:]=q0;d.qpos[qa[:n]]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    R=anchorR@d.xmat[anchor].reshape(3,3).T;translation=anchorP-R@d.xpos[anchor];d.qpos[:3]=R@q0[:3]+translation;d.qpos[3:7]=Rotation.from_matrix(R@rootR).as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    gaps=[]
    for a,b,i,j,kind in pairs:
        gap=.03 if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]+.03 else min(.03,float(mujoco.mj_geomDistance(m,d,a,b,.03,None)))
        gaps.append(gap);minimum_gap[kind]=min(minimum_gap[kind],gap)
    actual.qpos[:]=d.qpos;mujoco.mj_fwdPosition(physical,actual);maximum_source_motion=max(maximum_source_motion,float(np.linalg.norm(nodes(actual)[n:]-reference[n:],axis=1).max()))
    for ct in actual.contact:
        gs=[ct.geom1,ct.geom2];robot=[head<=physical.geom_bodyid[g]<head+21 for g in gs]
        if all(robot):kind=2
        elif sum(robot)==1:
            name=physical.geom(gs[0] if not robot[0] else gs[1]).name
            kind=3 if name=='floor' else 0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1 if name in ['pipe_sharp_out_03','pipe_straight_04'] else None
        else:kind=None
        if kind is not None:maximum_original_depth[kind]=max(maximum_original_depth[kind],-ct.dist)
    cache[key]=dict(t=float(t),q=q,pos=d.geom_xpos[geoms].copy(),R=d.geom_xmat[geoms].reshape(21,3,3).copy(),gaps=np.array(gaps));return cache[key]
certified=[];failed=[];minimum_floor_bound=np.inf
def certify(a,b,depth=0):
    global minimum_floor_bound
    A=evaluate(a);Z=evaluate(b);delta=abs(Z['q']-A['q']);motion=np.zeros(21)
    for i in range(n):
        geom=geoms[i];angle=np.arccos(np.clip((np.trace(A['R'][i].T@Z['R'][i])-1)/2,-1,1));endpoint=np.linalg.norm(Z['pos'][i]-A['pos'][i])+m.geom_rbound[geom]*angle
        length=(n-i)*.0645+np.linalg.norm(m.geom_pos[geom])+m.geom_rbound[geom];motion[i]=endpoint+length*float(delta[i:].sum())**2/8
    lower=np.array([max(A['gaps'][k],Z['gaps'][k])-motion[i]-(motion[j] if j>=0 else 0.)-2e-6 for k,(_,_,i,j,_) in enumerate(pairs)])
    limits=np.array([0. if kind=='self' else -.003 for *_,kind in pairs]);floor=np.minimum(A['pos'][:,2],Z['pos'][:,2])-m.geom_rbound[geoms]-motion;minimum_floor_bound=min(minimum_floor_bound,float(floor.min()))
    if (lower>=limits).all() and floor.min()>0:certified.append([a,b]);return
    if depth>=16 or b-a<1e-8:
        k=int(np.argmin(lower-limits));failed.append(dict(start=a,end=b,pair=pairs[k],bound_m=float(lower[k]),limit_m=float(limits[k])));return
    mid=(a+b)/2;certify(a,mid,depth+1);certify(mid,b,depth+1)
for a,b in zip(z['t'][:-1],z['t'][1:]):certify(float(a),float(b))
physical_safety=bool(max(maximum_original_depth[:3])<=.003 and maximum_original_depth[3]==0.)
report=dict(simulation_steps=0,role='Continuous convex-geometry sweep of fixed-M8 native hinge interpolation, with original-algorithm collision samples; not dynamics',
    passed=not failed and physical_safety,continuous_native_geometry_passed=not failed,original_collision_samples_within_3mm=physical_safety,
    evaluated_poses=len(cache),certified_subintervals=len(certified),certified_time_measure_s=float(sum(b-a for a,b in certified)),prefix_end_s=float(z['t'][-1]),
    fixed_source_node_motion_mm=maximum_source_motion*1000,original_depth_components=['source','receiver','self','floor'],maximum_original_depths_mm=(maximum_original_depth*1000).tolist(),
    minimum_evaluated_native_gaps_mm={k:v*1000 for k,v in minimum_gap.items()},minimum_conservative_floor_bound_m=minimum_floor_bound,signed_distance_uncertainty_m=2e-6,
    original_physical_enableflags=int(physical.opt.enableflags),private_native_distance_enableflags=int(m.opt.enableflags),
    method='Endpoint rigid-geometry movement plus serial-chain L*(sum|dq|)^2/8 remainder; signed convex distance is Lipschitz under this motion. Adaptive subdivision when needed. Original collision algorithm independently sampled at every evaluated pose.',
    limitation='The continuous bound applies to native convex geometry; original legacy collision values are samples and can show numerical seam artifacts. No force or dynamic stability proof.',failures=failed,wall_s=time.time()-start)
(O/'CONTINUOUS_CLEARANCE_CERTIFICATE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
