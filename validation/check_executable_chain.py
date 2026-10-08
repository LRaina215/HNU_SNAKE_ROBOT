"""Necessary derivative, registration and saved-start checks; no dynamics."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import numpy as np
from snake_control.executable_chain_transfer import ExecutableChainTransfer
from snake_control.hook_pull_review_support import restore_saved_state
from unified_curve_transfer_experiment import UnifiedSim


def check():
    out=ROOT/'experiments/unified_curve_transfer_20261007/EXECUTABLE_CHAIN_20261009';out.mkdir(exist_ok=True)
    sim=UnifiedSim(out);restore_saved_state(sim.m,sim.d,ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz');sim.stage='preflight';sim.snapshot()
    c=ExecutableChainTransfer(sim.m,sim.d);support=sim.rows[0]['cylindrical_contact_force_N']
    c.retain_observed_grip_clearance(sim.d,0.,support)
    guide,_,_=c.curve.reference(0.);c.reference_before_execution=guide.copy()
    evaluate,*_=c._problem(sim.d,support,guide,c.root_orientation,0.)
    q=sim.d.qpos[c.qa].copy();s=evaluate(q);analytic=s['jac'].copy();collision=s['gap_jac'].copy();active=s['gaps']<.04
    finite=np.zeros_like(analytic);finite_collision=np.zeros_like(collision);epsilon=1e-6
    for j in range(20):
        step=np.eye(20)[j]*epsilon
        s=evaluate(q+step);points=s['points'].copy();gaps=s['gaps'].copy()
        s=evaluate(q-step)
        finite[:,:,j]=(points-s['points'])/(2*epsilon);finite_collision[:,j]=(gaps-s['gaps'])/(2*epsilon)
    physical=sim.d.qpos.copy();ctrl=sim.d.ctrl.copy();physical_time=sim.d.time;start=time.perf_counter()
    target,info=c.command(sim.d,0.,sim.rows[0]['contact_force_N'],sim.rows[0]['torque'],support)
    report=dict(node_derivative_max_error_m_per_rad=float(np.abs(analytic-finite).max()),
        active_collision_derivative_max_error_normalized=float(np.abs(collision[active]-finite_collision[active]).max()) if active.any() else 0.,
        original_physical_qpos_unchanged=bool(np.array_equal(physical,sim.d.qpos)),first_ctrl_preserved=bool(np.array_equal(target,ctrl)),
        physical_time_unchanged=sim.d.time==physical_time,command_wall_s=time.perf_counter()-start,chain_fit=info['executable_chain'],
        collision_derivative_caveat='original mesh contact minima can switch features; collision derivative is local, candidate feasibility is checked on recomputed original contacts')
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    assert report['node_derivative_max_error_m_per_rad']<1e-6
    assert report['active_collision_derivative_max_error_normalized']<.02
    assert report['original_physical_qpos_unchanged'] and report['first_ctrl_preserved'] and report['physical_time_unchanged']
    assert info['executable_chain']['accepted_feasible_candidate']
    c.save_state(out/'preflight_controller_state.json',0.)
    print(json.dumps({k:v for k,v in report.items() if k!='chain_fit'}))


if __name__=='__main__':check()
