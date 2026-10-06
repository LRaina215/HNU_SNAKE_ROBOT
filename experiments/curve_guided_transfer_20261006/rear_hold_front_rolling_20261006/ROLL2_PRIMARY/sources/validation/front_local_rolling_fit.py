"""One justified geometry repair: consistent finite-chain local rolling fit.

The desired centerline keeps the original R95/P220 guide and finite initial
residual. Axial shift is a connector coordinate, not rear material feed.
No physics steps, radius/phase search, or state from a live simulation.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json,math,shutil,time
import numpy as np
import mujoco
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from front_local_rolling_geometry import BASE,ORIGINAL,write
from curve_local_path import LocalPathFit
from curve_guided_geometry import points
from snake_control.local_material_reference import BoundaryMaterialCurve
from snake_control.front_local_rolling import FrontLocalRollingReference


def material_twist(model,data):
    """Intrinsic cross-section angle against local curvature normal.

    Body z points along its own module. Internal tangents define a discrete
    curvature normal. Endpoint normals are parallel transported from the
    nearest internal frame; a one-hinge endpoint normal is degenerate.
    """
    ids=[model.body('head').id]+[model.body('link'+str(i)).id for i in range(1,20)]+[model.body('tail').id]
    R=data.xmat[ids].reshape(-1,3,3);t=R[:,:,2];y=R[:,:,1]
    n=np.zeros_like(t);n[1:-1]=t[2:]-t[:-2]
    n-=np.sum(n*t,axis=1)[:,None]*t
    length=np.linalg.norm(n,axis=1)
    n/=np.maximum(length[:,None],1e-12)
    for end,neighbor in [(0,1),(-1,-2)]:
        cross=np.cross(t[neighbor],t[end]);s=float(np.linalg.norm(cross));c=float(t[neighbor]@t[end])
        if s<1e-12:n[end]=n[neighbor]
        else:n[end]=Rotation.from_rotvec(cross/s*math.atan2(s,c)).apply(n[neighbor])
        length[end]=length[neighbor]
    b=np.cross(t,n)
    return np.arctan2(np.sum(y*b,axis=1),np.sum(y*n,axis=1)),length


def pipe_relative_twist(model,data):
    """Material y against nominal target radial frame, offline/score only.

    Unlike the internal curvature frame this asks how the module actually
    rolls relative to the pipe. It cannot be computed from encoders alone.
    """
    ids=[model.body('head').id]+[model.body('link'+str(i)).id for i in range(1,7)]
    R=data.xmat[ids].reshape(-1,3,3);t=R[:,:,2];y=R[:,:,1]
    p=points(model,data);centers=(p[:7]+p[1:8])/2
    n=centers-[1100.,0.,1000.];n[:,1]=0
    n-=np.sum(n*t,axis=1)[:,None]*t
    length=np.linalg.norm(n,axis=1);n/=np.maximum(length[:,None],1e-12);b=np.cross(t,n)
    return np.arctan2(np.sum(y*b,axis=1),np.sum(y*n,axis=1)),length


class ConsistentRollingFit:
    def __init__(self):
        self.fit=LocalPathFit();self.old=BoundaryMaterialCurve(ORIGINAL);self.x0=self.old.state(0.)
        self.fit.set(self.x0);self.p0=points(self.fit.m,self.fit.d).copy()
        self.anchor=self.fit.m.body('link12').id
        self.ap=self.fit.d.xpos[self.anchor].copy();self.aR=self.fit.d.xmat[self.anchor].reshape(3,3).copy()
        self.psi0,self.curv0=pipe_relative_twist(self.fit.m,self.fit.d)
        self.bodyids=[self.fit.m.body('head').id]+[self.fit.m.body('link'+str(i)).id for i in range(1,7)]
        self.R0=self.fit.d.xmat[self.bodyids].reshape(-1,3,3).copy()

    def state(self,q):
        x=self.x0.copy();x[6:18]=q[:12];self.fit.set(x)
        rotate=self.aR@self.fit.d.xmat[self.anchor].reshape(3,3).T
        x[:3]=(self.ap+rotate@(x[:3]/1000-self.fit.d.xpos[self.anchor]))*1000
        x[3:6]=(Rotation.from_matrix(rotate)*Rotation.from_rotvec(x[3:6])).as_rotvec()
        self.fit.set(x);return x

    def solve(self,phi,seed):
        if phi==0:return self.x0.copy(),np.r_[self.x0[6:18],0.,0.],dict(phase_deg=0.,exact_initial=True)
        qguide=self.x0[6:12]+self.old.prototype.joint_angles(0.,phi)[:6]-self.old.prototype.joint_angles(0.,0.)[:6]
        def residual(z):
            self.state(z);p=points(self.fit.m,self.fit.d)
            orbital=Rotation.from_rotvec([0.,z[13],0.]).as_matrix()
            desired=(self.p0[:8]-[1100,0,1000])@orbital.T+[1100,z[12],1000]
            psi,_=pipe_relative_twist(self.fit.m,self.fit.d)
            twist=(psi[:6]-self.psi0[:6]+phi+np.pi)%(2*np.pi)-np.pi
            # Centerline target and intrinsic phase fitted together. Axial
            # coordination is bounded and weakly minimized, not imposed feed.
            return np.r_[((p[:8]-desired)/4.).ravel(),twist/math.radians(1.5),
                .08*(z[:6]-qguide)/math.radians(5),.025*(z[:12]-seed[:12])/math.radians(5),
                z[12]/32.25*.05,z[13]/math.radians(5)*.05]
        low=np.r_[np.full(12,-math.radians(89)),-32.25,-math.radians(5)];high=-low
        ls=least_squares(residual,np.clip(seed,low,high),bounds=(low,high),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        def constraints(z):return self.fit.gaps(self.state(z))[:24]-.05
        if constraints(ls.x).min()<-.001:
            last=[time.monotonic()]
            def progress(z):
                if time.monotonic()-last[0]>15:
                    print(json.dumps(dict(phase_deg=float(np.degrees(phi)),solver='finite_joint_fit',physics_steps=0)),flush=True);last[0]=time.monotonic()
            units=np.r_[np.full(12,.1),10.,.1]
            out=minimize(lambda v:float(residual(ls.x+units*v)@residual(ls.x+units*v)),np.zeros(14),method='SLSQP',
                bounds=list(zip((low-ls.x)/units,(high-ls.x)/units)),
                constraints=[dict(type='ineq',fun=lambda v:constraints(ls.x+units*v))],callback=progress,
                options=dict(maxiter=100,ftol=1e-8,eps=1e-5))
            out.x=ls.x+units*out.x
        else:out=ls
        z=out.x;x=self.state(z);mujoco.mj_forward(self.fit.m,self.fit.d);p=points(self.fit.m,self.fit.d)
        psi,_=pipe_relative_twist(self.fit.m,self.fit.d);change=(psi[:6]-self.psi0[:6]+np.pi)%(2*np.pi)-np.pi
        desired=(self.p0[:8]-[1100,0,1000])@Rotation.from_rotvec([0.,z[13],0.]).as_matrix().T+[1100,z[12],1000]
        collisions=[dict(geom1=self.fit.m.geom(c.geom1).name,geom2=self.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in self.fit.d.contact if c.dist<0]
        row=dict(phase_deg=float(np.degrees(phi)),solver_success=bool(out.success),solver_message=str(out.message),
            axial_coordination_mm=float(z[12]),orbital_coordination_deg=float(np.degrees(z[13])),front_centerline_rms_mm=float(np.sqrt(np.mean(np.sum((p[:8]-desired)**2,axis=1)))),
            front_centerline_maximum_error_mm=float(np.linalg.norm(p[:8]-desired,axis=1).max()),
            pipe_relative_front_twist_change_deg=np.degrees(change).tolist(),minimum_conservative_gap_mm=float(self.fit.gaps(x).min()),
            modifiable_front_minimum_gap_mm=float(self.fit.gaps(x)[:24].min()),
            collisions=collisions,rear_reference_node_error_mm=float(abs(p[12:]-self.p0[12:]).max()),
            maximum_joint_deg=float(np.degrees(abs(x[6:])).max()),points_mm=p.tolist(),joint_angles_deg=np.degrees(x[6:]).tolist())
        return x.copy(),z.copy(),row


def main():
    registry=json.loads((BASE/'revision_registry.json').read_text(encoding='utf8'))
    if not registry['further_dynamics_authorized']:raise RuntimeError('Candidate closed; no further construction authorized')
    attempt=BASE/'geometry_attempt0_fixed_connection'
    if not attempt.exists():
        attempt.mkdir()
        for name in ['geometry_plan.json','geometry_progress.json','geometry_result.json','reference_partial.npz']:
            shutil.copy2(BASE/name,attempt/name)
    if (BASE/'reference_discrete.npz').exists():raise FileExistsError('Preserve completed finite fit')
    transported=BASE/'geometry_attempt2_internal_frame'
    if (BASE/'endpoint_frame_repair.json').exists() and not transported.exists():
        transported.mkdir()
        for name in ['geometry_progress.json','geometry_result.json','reference_partial.npz']:
            shutil.copy2(BASE/name,transported/name)
        write(BASE/'pipe_frame_definition.json',dict(physics_steps=0,
            reason='Internal discrete curvature twist mixes curvature shape with material rotation and duplicates head/internal neighbor; pipe-relative radial frame directly defines local roll',
            frame='Known nominal target radial normal perpendicular to each module tangent, material y atan2 against normal/binormal',
            truth_in_control=False,thresholds_unchanged=True,
            numerical_units='SLSQP variables scaled .1rad/10mm, same objective/constraints; fixes mixed-unit convergence, no model/control parameter change'))
    partial=BASE/'geometry_attempt1_endpoint_frame'
    if (BASE/'geometry_repair.json').exists() and not partial.exists():
        partial.mkdir()
        for name in ['geometry_progress.json','geometry_result.json','reference_partial.npz','geometry_repair.json']:
            shutil.copy2(BASE/name,partial/name)
        write(BASE/'endpoint_frame_repair.json',dict(physics_steps=0,
            issue='One-sided head curvature normal follows J1 axis and makes head twist identically constant; impossible all-module rolling target',
            correction='Parallel transport nearest internal curvature normal to each endpoint; intrinsic internal frames and target phase unchanged',
            no_geometric_or_dynamics_threshold_relaxation=True))
    write(BASE/'geometry_repair.json',dict(physics_steps=0,reason='1deg continuum increments and fixed link6 pose gave finite collisions and residual; no infeasibility proof',
        change='Jointly fit J1..J12 against unchanged centerline plus bounded axial connector coordination and intrinsic material twist -phi; rear link12 nominal pose/J13..20 unchanged',
        guide_radius_mm=95,guide_pitch_mm=220,front_phase_deg=[0,15],axial_coordinate_bound_mm=32.25,
        no_radius_initial_or_amplitude_search=True,collision_tolerances_unchanged=True,
        control_error_repair='Restore original nominal+retained-measured tracking relation; earlier last-measured omitted new nominal increment'))
    write(BASE/'final_geometry_construction.json',dict(physics_steps=0,
        reason='Axial-only desired nodes imposed fixed circumferential position absent from task; allow coupled pipe-axis orbit without changing radius/pitch',
        front_material_phase_deg=[0,15],front_orbital_coordinate_bound_deg=[-5,5],axial_bound_mm=[-32.25,32.25],
        orbit_bound_basis='5deg at R95 is 8.29mm, below one-sixth module length and one-third requested material roll; ideal no-slip estimate about1.58deg, not calibrated',
        ideal_no_slip='h=220/(2pi),Lh=sqrt(95^2+h^2),body_radius=29: axial=29*95/Lh*phi; orbital=-29*h/(95*Lh)*phi; geometric clue only, no traction proof',
        geometric_fitting_tolerance_mm=6.45,tolerance_status='chosen finite-guide approximation 10% module length; distinct from actual end-hold displacement gate, not a physical impossibility boundary',
        target='p_des=CORNER+Rot_y(orbital)*(p_initial-CORNER)+axial*e_y; same saved finite helix guide and R95/P220 design',
        rear='J13..20 nominal and private link12 pose fixed; no physical root command',last_construction_repair=True,
        if_failed='stop geometry implementation, do not claim method impossible or run dynamics'))
    plan=json.loads((BASE/'geometry_plan.json').read_text(encoding='utf8'))
    plan.update(front='Joint finite J1..J12 fit for pipe-relative material roll -phi and constant-R/P guide; frontJ1..6, connectorJ7..12',
        connector='Finite link12 nominal anchor; front desired nodes allowed axial +/-32.25mm and orbital +/-5deg; six transition joints coordinate positions/directions',
        actual_phase='Material y angle against pipe radial normal perpendicular to own module tangent; truth diagnostic only; internal transported curvature twist independently reported')
    write(BASE/'geometry_plan.json',plan)
    fit=ConsistentRollingFit();phases=np.radians(np.arange(0,15.001,1));states=[];rows=[];shifts=[];orbits=[];seed=np.r_[fit.x0[6:18],0.,0.]
    for phi in phases:
        x,seed,row=fit.solve(float(phi),seed);states.append(x);rows.append(row);shifts.append(seed[12]);orbits.append(seed[13])
        write(BASE/'geometry_progress.json',dict(physics_steps=0,rows=rows))
        np.savez_compressed(BASE/'reference_partial.npz',phase_rad=phases[:len(states)],joint_rad=np.array(states)[:,6:18],axial_mm=shifts,orbital_rad=orbits)
        print(json.dumps({k:row.get(k) for k in ['phase_deg','axial_coordination_mm','orbital_coordination_deg','front_centerline_rms_mm','pipe_relative_front_twist_change_deg','minimum_conservative_gap_mm']}),flush=True)
        if row.get('collisions') or row.get('front_centerline_maximum_error_mm',0)>6.45 or row.get('modifiable_front_minimum_gap_mm',.05)<.049:
            write(BASE/'geometry_result.json',dict(status='NOT_READY',physics_steps=0,failure=row));return
    np.savez_compressed(BASE/'reference_discrete.npz',phase_rad=phases,joint_rad=np.array(states)[:,6:18],axial_mm=shifts,orbital_rad=orbits)
    ref=FrontLocalRollingReference(BASE/'reference_discrete.npz',ORIGINAL);failures=[];rear=length=maximum=0.
    for phi in np.radians(np.arange(0,15.001,.05)):
        x=ref.state(float(phi));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d);p=points(fit.fit.m,fit.fit.d)
        negative=[dict(geom1=fit.fit.m.geom(c.geom1).name,geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        if negative:failures.append(dict(phase_deg=float(np.degrees(phi)),collisions=negative))
        rear=max(rear,float(abs(p[12:]-fit.p0[12:]).max()));length=max(length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
        maximum=max(maximum,float(np.degrees(abs(x[6:])).max()))
    tt=np.arange(541)*.05;u=np.clip((tt-5)/12,0,1);phase=math.radians(15)*u*u*(3-2*u)
    stage=np.where(tt<5,'entry',np.where(tt<17,'front_rolling','end_hold'))
    qq=np.array([ref.joint_angles(float(v)) for v in phase]);rate=float(np.degrees(abs(np.diff(qq,axis=0))).max()/.05)
    cmdcollisions=[]
    for t,phi in zip(tt,phase):
        fit.fit.set(ref.state(float(phi)));mujoco.mj_forward(fit.fit.m,fit.fit.d)
        if any(c.dist<0 for c in fit.fit.d.contact):cmdcollisions.append(float(t))
    twist=np.array(rows[-1]['pipe_relative_front_twist_change_deg']);twistpass=bool(np.all(twist<-7.5) and abs(twist.mean()+15)<3)
    passed=not failures and not cmdcollisions and rate<=9 and maximum<89.001 and rear<1e-6 and length<1e-6 and twistpass
    result=dict(status='READY_FOR_INTERFACE' if passed else 'NOT_READY',physics_steps=0,key_poses=rows,
        interpolation_failures=failures,command_collision_times=cmdcollisions,maximum_command_rate_deg_s=rate,
        maximum_joint_deg=maximum,rear_reference_node_maximum_error_mm=rear,maximum_length_error_mm=length,
        nominal_intrinsic_rolling_pass=twistpass,rear_nominal_joint_error_rad=float(abs(qq[:,12:]-fit.x0[18:]).max()),
        finite_sampling='.05deg and all commands, not mathematical continuum proof',old_V3_path='NOT_READY')
    write(BASE/'geometry_result.json',result)
    if passed:
        shutil.copy2(BASE/'reference_discrete.npz',BASE/'reference_ready.npz')
        np.savez_compressed(BASE/'command_schedule.npz',time_s=tt,phase_rad=phase,stage=stage)
        write(BASE/'path_gate.json',dict(geometry_and_command_timing_pass=True,total_s=27.,old_V3_path='NOT_READY'))
    print(json.dumps({k:result[k] for k in ['status','maximum_command_rate_deg_s','nominal_intrinsic_rolling_pass','rear_reference_node_maximum_error_mm']}),flush=True)


if __name__=='__main__':main()
