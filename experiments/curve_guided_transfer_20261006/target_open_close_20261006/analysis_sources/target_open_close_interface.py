"""Necessary interface/shape checks without physics evolution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from target_open_close_geometry import BASE,OLD,ORIGINAL,write
from snake_control.target_open_close_reference import TargetOpenCloseReference
from snake_control.target_open_close_control import TargetOpenCloseController
from snake_control.executable_grasp_state import EquilibriumGraspController
from snake_control.local_material_reference import BoundaryMaterialCurve
from curve_guided_geometry import points,MODEL


def own_cylinder(p):
    """Diagnostic free cylinder fit; radius is distinct from pipe-axis distance.

    Seven/eight finite centerline nodes are a short helix. Jacobian conditioning
    and residual are reported; this is not an unambiguous intrinsic radius.
    """
    p=np.asarray(p,float);center=p.mean(0);v=p-center
    def axis(z):
        a=np.array([z[0],1.,z[1]]);return a/np.linalg.norm(a)
    def residual(z):
        a=axis(z);c=np.array([z[2],0.,z[3]])
        vv=v-c;cross=vv-np.outer(vv@a,a)
        return np.linalg.norm(cross,axis=1)-z[4]
    rr=np.mean(np.linalg.norm(v[:,[0,2]],axis=1))
    result=least_squares(residual,[0.,0.,0.,0.,max(10,rr)],
        bounds=([-4,-4,-400,-400,10],[4,4,400,400,500]),max_nfev=200,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    z=result.x;sv=np.linalg.svd(result.jac,compute_uv=False)
    condition=float(sv[0]/max(sv[-1],1e-12))
    rms=float(np.sqrt(np.mean(residual(z)**2)))
    a=axis(z);c=center+np.array([z[2],0,z[3]])
    radial_offset=float(np.linalg.norm((c-[1100,center[1],1000])[[0,2]]))
    return dict(fitted_own_radius_mm=float(z[4]),axis_world=a.tolist(),
        axis_angle_to_pipe_deg=float(np.degrees(np.arccos(np.clip(abs(a[1]),0,1)))),
        axis_offset_at_front_mean_y_mm=radial_offset,residual_rms_mm=rms,jacobian_condition=condition,
        reliable=bool(result.success and rms<8 and condition<1000 and not np.any(abs(z[:2])>3.99)),
        limitation='short variable-radius discrete helix; fit is descriptive, not independent pipe sensing')


def own_helix(p):
    """Ordered spatial helix fit; translates/rotates freely, unlike axis distance.

    Preserve axial progression as well as radius. A cylinder-only residual can
    fit a tilted cylinder to the same six nodes and give a misleading radius.
    The fitting window/model is still a diagnostic approximation.
    """
    p=np.asarray(p,float);xy=(p-[1100,0,1000])[:,[0,2]]
    solution=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy*xy).sum(1),rcond=None)[0]
    center=solution[:2];radius=float(np.sqrt(max(1,solution[2]+center@center)))
    a=np.array([0.,1.,0.]);u=np.array([xy[0,0]-center[0],0.,xy[0,1]-center[1]])
    u/=np.linalg.norm(u);v=np.cross(a,u)
    rot=Rotation.from_matrix(np.column_stack((u,v,a)))
    phase=np.unwrap(np.arctan2((p-p.mean(0))@v,(p-p.mean(0))@u))
    phase_step=float(np.mean(np.diff(phase)));axial_step=float(np.mean(np.diff(p[:,1])))
    origin=np.array([1100+center[0],p[0,1],1000+center[1]])
    idx=np.arange(len(p))
    def residual(w):
        rr=Rotation.from_rotvec(w[3:6]).as_matrix()
        pred=w[:3]+w[6]*np.cos(idx*w[7])[:,None]*rr[:,0]+w[6]*np.sin(idx*w[7])[:,None]*rr[:,1]+idx[:,None]*w[8]*rr[:,2]
        return (pred-p).ravel()
    initial=np.r_[origin,rot.as_rotvec(),radius,phase_step,axial_step]
    result=least_squares(residual,initial,max_nfev=300,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    w=result.x;a=Rotation.from_rotvec(w[3:6]).as_matrix()[:,2]
    units=np.array([10,10,10,.1,.1,.1,10,.1,10])
    sv=np.linalg.svd(result.jac*units[None,:],compute_uv=False)
    condition=float(sv[0]/max(sv[-1],1e-12));rms=float(np.sqrt(np.mean(residual(w)**2)))
    mean_y=float(p[:,1].mean());axispoint=w[:3]
    if abs(a[1])>1e-6:axispoint=axispoint+(mean_y-axispoint[1])/a[1]*a
    offset=float(np.linalg.norm((axispoint-[1100,mean_y,1000])[[0,2]]))
    sigma2=float(np.sum(residual(w)**2)/max(1,len(p)*3-9))
    covariance=np.linalg.pinv(result.jac.T@result.jac)*sigma2
    radius_std=float(np.sqrt(max(0,covariance[6,6])))
    valid=bool(result.success and 20<w[6]<300 and rms<8 and condition<300 and radius_std<35)
    return dict(fitted_own_radius_mm=float(w[6]),axis_world=a.tolist(),
        axis_angle_to_pipe_deg=float(np.degrees(np.arccos(np.clip(abs(a[1]),0,1)))),
        axis_offset_at_front_mean_y_mm=offset,residual_coordinate_rms_mm=rms,
        jacobian_scaled_condition=condition,linearized_radius_std_mm=radius_std,
        phase_step_rad=float(w[7]),axial_step_mm=float(w[8]),reliable=valid,
        conditional_pipe_transverse_circle_radius_mm=radius,
        conditional_circle_center_offset_mm=float(np.linalg.norm(center)),
        limitation='six ordered finite nodes, uniform helix approximation; axis/radius fit does not establish grip or topology')


def main():
    if (BASE/'interface_check.json').exists():
        oldcheck=json.loads((BASE/'interface_check.json').read_text(encoding='utf8'))
        if oldcheck.get('interface_and_nominal_opening_pass'):raise FileExistsError('Successful interface already recorded')
        write(BASE/'interface_cylinder_only_attempt0.json',oldcheck)
    reference=TargetOpenCloseReference(BASE/'reference_ready.npz',ORIGINAL)
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    with np.load(OLD/'R2_C0/trace.npz') as z:data.qpos[:]=z['qpos'][0]
    mujoco.mj_kinematics(model,data);registered=data.body('head').xmat.reshape(3,3).copy()
    def fk(q):
        data.qpos[:]=0;data.qpos[3]=1;data.qpos[qadr]=q;mujoco.mj_kinematics(model,data)
        return points(model,data)
    config=json.loads((OLD/'R2_SPEED3_PAIR_C0/config.json').read_text(encoding='utf8'))
    with np.load(BASE/'initial_boundary_pose.npz') as initial:
        target=initial['target'].copy();measured=initial['qpos'][qadr].copy()
    old=EquilibriumGraspController(BoundaryMaterialCurve(ORIGINAL),config,target,fk,registered)
    new=TargetOpenCloseController(reference,config,target,fk,registered)
    qo,lo=old.compute(measured,0.,45.25,.05);qn,ln=new.compute(measured,0.,45.25,.05)
    samples=[]
    for a in [0.,1e-8,1e-6,.001,.01]:
        c=TargetOpenCloseController(reference,config,target,fk,registered)
        q,l=c.compute(measured,a,45.25,.05)
        samples.append(dict(action_mm=a,first_target_difference_deg=float(np.degrees(abs(q-qn)).max())))
    geometry=json.loads((BASE/'geometry_result.json').read_text(encoding='utf8'))
    rows=[]
    for row in geometry['key_poses']:
        p=np.asarray(row['points_mm']);feed=row['feed_mm']
        own=own_helix(p[:6])
        rows.append(dict(action_mm=row['action_mm'],feed_mm=feed,opening_mm=row['opening_mm'],
            nominal_front_free_helix=own,front_modules='M0..M4 centerline nodes0..5 fixed diagnostic window; all within full R130 plateau during opening'))
    start=rows[0]['nominal_front_free_helix'];openrow=rows[1]['nominal_front_free_helix']
    increase=openrow['fitted_own_radius_mm']-start['fitted_own_radius_mm']
    eq=float(abs(qn-qo).max());rerr=float(abs(np.array(ln['retained_shape_rad'])-lo['retained_shape_rad']).max())
    passed=eq<1e-12 and rerr<1e-12 and samples[2]['first_target_difference_deg']<.00001 and increase>=20
    write(BASE/'interface_check.json',dict(physics_steps=0,starting_target_equivalence_rad=eq,
        starting_state_equivalence_rad=rerr,complete_starting_branch_proof='action0 directly delegates complete R2 with identical original reference/root/tangent for any observation/ACK',
        boundary_samples=samples,nominal_opening_free_radius_increase_mm=increase,
        nominal_shape_diagnostics=rows,interface_and_nominal_opening_pass=bool(passed),
        diagnostic_repair='cylinder-only fit ignored ordered axial progression, produced wrong tilted solutions; archived attempt0; uniform ordered spatial helix fit used, uncertainty retained',
        distinction='free fitted cylinder radius and its offset/angle versus known target pipe are separate diagnostic quantities',
        actual_opening_gate='During full-open interval fit reliable and actual own-radius increase>=20mm; closing own radius decreases>=15mm from open interval. Shape-only, not grip',
        full_V3_path='NOT_READY'))
    print(json.dumps(dict(pass_=bool(passed),starting_target_error=eq,nominal_opening_increase_mm=increase,shape_diagnostics=rows)),flush=True)


if __name__=='__main__':main()
