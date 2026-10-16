"""No integration: t0 continuity, input boundary, native chain and post route."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,argparse
from physics import *
from direct_controller import DirectController
from post_capture_curve import PostCaptureCurve

def run(args):
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'))
    if args.released_hinges is not None:reg['direct_extension']['released_hinges']=args.released_hinges;reg['kinematic_solver']['release_hinges']=args.released_hinges
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial if args.initial else BASE/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    q0=d.qpos.copy();ctrl0=d.ctrl.copy();control=DirectController(m,d,reg,'DIRECTION',True);raw,info=control.command(d,0.)
    integral=control.planner.coordinates.hinges(info['curvature_field']);fieldpeak=float(np.linalg.norm(info['curvature_field'],axis=1).max()*1000)
    checks=dict(t0_native_command_continuity=bool(np.max(abs(raw-ctrl0))<1e-12),t0_no_actual_pose_write=bool(np.array_equal(q0,d.qpos)),
        native_integral=bool(np.max(abs(integral-raw))<1e-12),t0_source_feed_and_roll_zero=control.S==0. and info['source_commanded_feed_m']==0. and info['source_commanded_roll_rad']==0.,
        t0_no_probe=bool(np.array_equal(info['probe_curvature_per_m'],np.zeros(2))),initial_curvature_within24_per_m=fieldpeak<=24.+1e-10)
    hold,hold_info=control.command(d,.025)
    checks.update(prescribed_source_hold_command_unchanged=bool(abs(hold-ctrl0).max()<1e-12),source_hold_no_actual_pose_write=bool(np.array_equal(q0,d.qpos)),
        alpha_source_guard_starts_M8=hold_info['source_guard_material_start']==8,source_hold_front_unloading_zero=bool(np.array_equal(control.front_relief,np.zeros(2))))
    from contact_sensor import read_sensor,classified_observation
    sensor=read_sensor(m,d);observed=classified_observation(sensor,control.estimator.geometry,reg['source_known']);support=control.support(sensor,0.,observed)
    report=dict(version=reg['version'],checks=checks,passed=all(checks.values()),simulation_steps=0,source_hold_native_error_deg=float(abs(raw-ctrl0).max()*180/np.pi),initial_curvature_peak_per_m=fieldpeak,
        initial_file=str(args.initial if args.initial else BASE/'GIVEN_INITIAL.npz'),initial_source_support=support,initial_supply_adequate=bool(control.adequate_supply(support,reg['robot_weight_N'])),
        scope='Static controller entry and native chain only. Does not validate physical source stability, acquisition, wrapping, transport or roll.')
    (Path(args.output) if args.output else BASE/'CONTROLLER_ENTRY_AUDIT.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else list(x) if isinstance(x,set) else str(x))+'\n');print(json.dumps(report,indent=2,default=str))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial');ap.add_argument('--released-hinges',type=int);ap.add_argument('--output');run(ap.parse_args())
