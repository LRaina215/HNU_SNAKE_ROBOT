"""Prescribed source-positive helix at a suitable exit phase; no dynamics."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json
from scipy.spatial.transform import Rotation
from physics import *

def run():
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    m=mujoco.MjModel.from_xml_path(str(MODEL));private=mujoco.MjModel.from_xml_path(str(MODEL));private.opt.enableflags|=64;d=mujoco.MjData(m);pvt=mujoco.MjData(private);z=np.load(BASE/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);q0=d.qpos.copy();ctrl0=d.ctrl.copy();qa=m.jnt_qposadr[m.actuator_trnid[:,0]];head=m.body('head').id;oldp=nodes(d).copy();oldR=d.xmat[head].reshape(3,3).copy();F=np.diag([1.,-1.,1.]);J=np.diag([1.,-1.,1.]);origin=np.array([0.,0.,1.]);geoms=[int(m.body_geomadr[head+i]) for i in range(21)];targets=[m.geom(n).id for n in ['pipe_sharp_out_03','pipe_straight_04']];candidates=[]
    for phase_deg in [-45.,-30.,-15.]:
        rx=Rotation.from_rotvec(np.radians(phase_deg+90)*np.array([1.,0.,0.])).as_matrix();rootR=rx@F@oldR@J;pose=q0.copy();pose[:3]=origin+rx@F@(q0[:3]-origin);pose[3:7]=Rotation.from_matrix(rootR).as_quat()[[3,0,1,2]];pose[qa[::2]]*=-1
        pvt.qpos[:]=pose;mujoco.mj_kinematics(private,pvt);p=nodes(pvt);c=.5*(p[:-1]+p[1:]);headX=oldp[0,0];upper=headX+1.095-float((c[:,0]+c[:,1]).max())
        def assign(x,physical=False):
            posecopy=pose.copy();posecopy[0]+=x-headX
            if physical:
                d.qpos[:]=posecopy;d.qvel[:]=0.;d.ctrl[:]=ctrl0;d.ctrl[::2]*=-1;d.qacc_warmstart[:]=0.;mujoco.mj_forward(m,d)
            else:pvt.qpos[:]=posecopy;mujoco.mj_kinematics(private,pvt)
            return posecopy
        def receiving_gap(x):
            assign(x);return min(min(.03,float(mujoco.mj_geomDistance(private,pvt,a,b,.03,None))) for a in geoms for b in targets)
        lo=.88;hi=upper
        for _ in range(24):
            mid=.5*(lo+hi)
            if receiving_gap(mid)>=.002:lo=mid
            else:hi=mid
        selected=lo;original_depth=None
        for offset in np.arange(0,.0121,.001):
            x=lo-offset;assign(x,True);depth=np.zeros(6)
            for ct in d.contact:
                gs=[ct.geom1,ct.geom2];robot=[head<=m.geom_bodyid[g]<head+21 for g in gs]
                if all(robot):depth[4]=max(depth[4],-ct.dist);continue
                if sum(robot)!=1:continue
                name=m.geom(gs[0] if not robot[0] else gs[1]).name;names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor']
                if name in names:k=names.index(name);depth[5 if k==4 else k]=max(depth[5 if k==4 else k],-ct.dist)
            if original_depth is None or max(depth[:2])<max(original_depth[:2]):selected=x;original_depth=depth.copy()
            if max(depth[:2])<=.0015:break
        assign(selected,True);p=nodes(d);c=.5*(p[:-1]+p[1:]);rho=np.linalg.norm(c[:,1:]-[0,1],axis=1);T=(p[0]-p[1])/.0645
        out=BASE.parent/f'INITIAL_V5_POSITIVE_PHASE_{int(abs(phase_deg)):02d}';out.mkdir(exist_ok=True);state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),points=p)
        report=dict(role='Prescribed all-source positive-handed source coil, unreviewed',simulation_steps=0,phase_deg=phase_deg,head_tip_m=p[0].tolist(),head_T=T.tolist(),source_axial_chirality='positive about source+X; same physical chirality as desired decreasing-phi receiver wrap',
            all21_source_envelope=bool(np.all((rho>.065)&(rho<.13)&(c[:,0]+c[:,1]<1.1))),original_depths_mm=(original_depth*1000).tolist(),native_receiver_initial_gap_mm=float(receiving_gap(selected)*1000),joint_peak_deg=float(abs(d.qpos[qa]).max()*180/np.pi),qacc_max=float(abs(d.qacc).max()),source_M14to20_axial_X_m=d.xpos[head+14:head+21,0].tolist(),new_dynamics_starts=0,current_selected_given_initial_unchanged=True)
        (out/'CANDIDATE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');candidates.append(report);print(json.dumps(report),flush=True)
    (BASE.parent/'POSITIVE_SOURCE_CANDIDATES.json').write_text(json.dumps(candidates,indent=2)+'\n')
if __name__=='__main__':run()
