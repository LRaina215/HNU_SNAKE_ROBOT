"""Local front roll and finite six-joint connector with fixed rear nominal."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import math
import csv
import shutil
import numpy as np
import mujoco
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from curve_local_path import LocalPathFit
from curve_guided_geometry import points
from snake_control.local_material_reference import BoundaryMaterialCurve
from snake_control.front_local_rolling import FrontLocalRollingReference

PARENT=ROOT/'experiments/curve_guided_transfer_20261006'
OLD=PARENT/'stage_continuation_20261006'
BASE=PARENT/'rear_hold_front_rolling_20261006'
ORIGINAL=OLD/'local_path_boundary_speed3/reference_ready.npz'


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class RollingFit:
    def __init__(self):
        self.fit=LocalPathFit();self.old=BoundaryMaterialCurve(ORIGINAL);self.x0=self.old.state(0.)
        self.fit.set(self.x0);self.p0=points(self.fit.m,self.fit.d).copy()
        self.anchor=self.fit.m.body('link12').id;self.front=self.fit.m.body('link6').id
        self.anchor_p=self.fit.d.xpos[self.anchor].copy();self.anchor_R=self.fit.d.xmat[self.anchor].reshape(3,3).copy()
        self.front_p=self.fit.d.xpos[self.front].copy();self.front_R=self.fit.d.xmat[self.front].reshape(3,3).copy()

    def state(self,phase,connector):
        x=self.x0.copy()
        x[6:12]+=self.old.prototype.joint_angles(0.,phase)[:6]-self.old.prototype.joint_angles(0.,0.)[:6]
        x[12:18]=connector
        self.fit.set(x)
        rotate=self.anchor_R@self.fit.d.xmat[self.anchor].reshape(3,3).T
        x[:3]=(self.anchor_p+rotate@(x[:3]/1000-self.fit.d.xpos[self.anchor]))*1000
        x[3:6]=(Rotation.from_matrix(rotate)*Rotation.from_rotvec(x[3:6])).as_rotvec();self.fit.set(x)
        return x

    def solve(self,phase,seed):
        if phase==0:return self.x0.copy(),dict(phase_deg=0.,exact_initial=True)
        desired_R=Rotation.from_rotvec(-phase*self.front_R[:,2]).as_matrix()@self.front_R
        def errors(q):
            x=self.state(phase,q)
            pos=(self.fit.d.xpos[self.front]-self.front_p)*1000
            rot=(Rotation.from_matrix(self.fit.d.xmat[self.front].reshape(3,3))*Rotation.from_matrix(desired_R).inv()).as_rotvec()
            return pos,rot
        def residual(q):
            pos,rot=errors(q)
            return np.r_[pos,64.5*rot]
        ls=least_squares(residual,seed,bounds=(-math.radians(89),math.radians(89)),max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        def constraints(q):
            x=self.state(phase,q)
            # Only modules0..11 can move; exact fixed downstream initial
            # conservative polygon gap deficit cannot be solved with these DOFs.
            return np.r_[self.fit.gaps(x)[:24]-.05,1.-np.linalg.norm(errors(q)[0]),math.radians(2)-np.linalg.norm(errors(q)[1])]
        if constraints(ls.x).min()<-.001:
            result=minimize(lambda q:float(residual(q)@residual(q)),ls.x,method='SLSQP',
                bounds=[(-math.radians(89),math.radians(89))]*6,constraints=[dict(type='ineq',fun=constraints)],
                options=dict(maxiter=100,ftol=1e-9,eps=1e-5))
            q=result.x;solver=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
        else:q=ls.x;solver=dict(success=bool(ls.success),message=str(ls.message),iterations=int(ls.nfev))
        x=self.state(phase,q);mujoco.mj_forward(self.fit.m,self.fit.d);p=points(self.fit.m,self.fit.d)
        contacts=[dict(body1=self.fit.m.body(int(self.fit.m.geom_bodyid[c.geom1])).name,
            body2=self.fit.m.body(int(self.fit.m.geom_bodyid[c.geom2])).name,distance_mm=float(c.dist*1000)) for c in self.fit.d.contact if c.dist<0]
        pos,rot=errors(q)
        row=dict(phase_deg=float(np.degrees(phase)),solver=solver,front_connection_position_error_mm=float(np.linalg.norm(pos)),
            front_connection_orientation_error_deg=float(np.degrees(np.linalg.norm(rot))),
            maximum_joint_deg=float(np.degrees(abs(x[6:])).max()),collisions=contacts,
            rear_reference_node_error_mm=float(abs(p[12:]-self.p0[12:]).max()),
            front_reference_node_maximum_change_mm=float(np.linalg.norm(p[:7]-self.p0[:7],axis=1).max()),
            points_mm=p.tolist(),joint_angles_deg=np.degrees(x[6:]).tolist())
        return x.copy(),row


def main():
    BASE.mkdir(exist_ok=False)
    with (PARENT/'trial_ledger.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if len(rows)!=17:raise RuntimeError('Expected preserved17 historical trials')
    shutil.copy2(PARENT/'trial_ledger.csv',BASE/'historical_17_trial_ledger.csv')
    shutil.copy2(OLD/'initial_boundary_pose.npz',BASE/'initial_boundary_pose.npz')
    shutil.copy2(ORIGINAL,BASE/'original_reference.npz')
    with np.load(BASE/'initial_boundary_pose.npz') as z:
        changed={k:z[k].copy() for k in z.files};changed['qpos'][0]+=.001
        np.savez_compressed(BASE/'initial_boundary_pose_perturb.npz',**changed)
    with (BASE/'trial_ledger.csv').open('w',encoding='utf8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
    write(BASE/'revision_registry.json',dict(status='OFFLINE',prior_dynamics=17,new_dynamics=0,total_dynamics=17,
        new_budget=4,total_budget=21,further_dynamics_authorized=True,maximum_targeted_revisions=1,
        freeze='root x+1mm before first step; matched phase0 C0 required',old_V3_path='NOT_READY'))
    write(BASE/'geometry_plan.json',dict(physics_steps=0,front_joints=[1,6],connector_joints=[7,12],rear_joints=[13,20],
        range_basis='J1..J6 integration windows wholly target helix; six connector DOFs bridge mixed arc/straight and first source wrap; J13..J20 pure source helix',
        front_phase_deg=[0,15],direction='positive curvature phase; material-frame twist about forward tangent negative; ideal no-slip tendency +targetY, actual advance separately scored',
        material_feed_mm=0,rear_active_phase=0,radius_mm=95,pitch_mm=220,
        front='saved finite nominal + analytical constant-R/P curvature-phase change on J1..J6',
        connector='finite IK fixes front link6 position and its tangent, supplies local axial-frame roll -phase while fixed rear link12 nominal pose remains; no physical pose fixing',
        front_connection_position_tolerance_mm=1.,front_connection_orientation_tolerance_deg=2.,
        timing=dict(entry_s=5,rolling_s=12,end_hold_s=10),phase_peak_rate_deg_s=1.875,total_s=27,
        anchor_scoring='actual modulesM13..M20 maximum displacement<=6.45mm entire action, azimuth change<=5deg, distributed >=2 actual source contacts',
        stop_holding='original last1s two-side contact/speed screen; last2s drift<=6.45mm, last5s speed<5mm/s; target still inside front projection and distributed contacts',
        actual_phase='cross-section Y-axis twist relative to locally reconstructed normal/binormal, using actual module poses; rigid world rotation cancels; not the phase clock',
        freeze='root x +1mm before first physical step, same saved nominal/ACK, matched C0 first',no_parameter_scan=True))
    fit=RollingFit();phases=np.radians(np.arange(0,15.001,1.));states=[];records=[];seed=fit.x0[12:18].copy()
    for phi in phases:
        x,row=fit.solve(float(phi),seed);seed=x[12:18];states.append(x);records.append(row)
        np.savez_compressed(BASE/'reference_partial.npz',phase_rad=phases[:len(states)],connector_rad=np.array(states)[:,12:18])
        write(BASE/'geometry_progress.json',dict(physics_steps=0,rows=records))
        print(json.dumps({k:row.get(k) for k in ['phase_deg','front_connection_position_error_mm','front_connection_orientation_error_deg','front_reference_node_maximum_change_mm']}),flush=True)
        if row.get('collisions') or row.get('front_connection_position_error_mm',0)>1.001 or row.get('front_connection_orientation_error_deg',0)>2.001:
            write(BASE/'geometry_result.json',dict(status='NOT_READY',physics_steps=0,failure=row));return
    np.savez_compressed(BASE/'reference_discrete.npz',phase_rad=phases,connector_rad=np.array(states)[:,12:18])
    ref=FrontLocalRollingReference(BASE/'reference_discrete.npz',ORIGINAL)
    failures=[];rearerror=0.;lengtherror=0.;maximum=0.
    for phi in np.radians(np.arange(0,15.001,.05)):
        x=ref.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d);p=points(fit.fit.m,fit.fit.d)
        contacts=[dict(body1=fit.fit.m.body(int(fit.fit.m.geom_bodyid[c.geom1])).name,
            body2=fit.fit.m.body(int(fit.fit.m.geom_bodyid[c.geom2])).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        if contacts:failures.append(dict(phase_deg=float(np.degrees(phi)),contacts=contacts))
        rearerror=max(rearerror,float(abs(p[12:]-fit.p0[12:]).max()))
        lengtherror=max(lengtherror,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
        maximum=max(maximum,float(np.degrees(abs(x[6:])).max()))
    tt=np.arange(541)*.05;phase=[];stage=[]
    for t in tt:
        u=np.clip((t-5)/12,0,1);phase.append(math.radians(15)*u*u*(3-2*u));stage.append('entry' if t<5 else 'front_rolling' if t<17 else 'end_hold')
    qq=np.array([ref.joint_angles(p) for p in phase]);rate=float(np.degrees(abs(np.diff(qq,axis=0))).max()/.05)
    commandcollisions=[]
    for t,phi in zip(tt,phase):
        fit.fit.set(ref.state(phi));mujoco.mj_forward(fit.fit.m,fit.fit.d)
        if any(c.dist<0 for c in fit.fit.d.contact):commandcollisions.append(float(t))
    passed=not failures and not commandcollisions and rate<=9 and maximum<89.001 and rearerror<1e-6 and lengtherror<1e-6
    result=dict(status='READY_FOR_INTERFACE' if passed else 'NOT_READY',physics_steps=0,
        interpolation_failures=failures,command_collision_times=commandcollisions,maximum_command_rate_deg_s=rate,
        maximum_joint_deg=maximum,rear_reference_node_maximum_error_mm=rearerror,maximum_length_error_mm=lengtherror,
        rear_nominal_joint_error_rad=float(abs(qq[:,12:]-fit.x0[18:]).max()),key_poses=records,
        finite_sampling='.05deg plus all16knots/all commands, not continuous proof',old_V3_path='NOT_READY')
    write(BASE/'geometry_result.json',result)
    if passed:
        shutil.copy2(BASE/'reference_discrete.npz',BASE/'reference_ready.npz')
        np.savez_compressed(BASE/'command_schedule.npz',time_s=tt,phase_rad=phase,stage=stage)
        write(BASE/'path_gate.json',dict(geometry_and_command_timing_pass=True,total_s=27.,old_V3_path='NOT_READY'))
    print(json.dumps({k:result[k] for k in ['status','maximum_command_rate_deg_s','rear_reference_node_maximum_error_mm','maximum_joint_deg']}),flush=True)


if __name__=='__main__':main()
