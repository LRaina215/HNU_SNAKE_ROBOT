"""Necessary interface check and full phase-zero equivalence, zero physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import numpy as np
import mujoco
from curve_guided_geometry import MODEL,points
from front_local_rolling_geometry import BASE,OLD,ORIGINAL,write
from snake_control.front_local_rolling import FrontLocalRollingReference,FrontLocalRollingController
from snake_control.executable_grasp_state import EquilibriumGraspController
from snake_control.local_material_reference import BoundaryMaterialCurve


def main():
    gate=json.loads((BASE/'path_gate.json').read_text(encoding='utf8'))
    if not gate['geometry_and_command_timing_pass']:raise RuntimeError('Geometry not admitted')
    reference=FrontLocalRollingReference(BASE/'reference_ready.npz',ORIGINAL)
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
    with np.load(OLD/'R2_C0/trace.npz') as z:d.qpos[:]=z['qpos'][0]
    mujoco.mj_kinematics(m,d);registered=d.body('head').xmat.reshape(3,3).copy()
    def fk(q):
        d.qpos[:]=0;d.qpos[3]=1;d.qpos[qadr]=q;mujoco.mj_kinematics(m,d);return points(m,d)
    config=json.loads((OLD/'R2_SPEED3_PAIR_C0/config.json').read_text(encoding='utf8'))
    with np.load(BASE/'initial_boundary_pose.npz') as z:target=z['target'].copy();measured=z['qpos'][qadr].copy()
    old=EquilibriumGraspController(BoundaryMaterialCurve(ORIGINAL),config,target,fk,registered)
    new=FrontLocalRollingController(reference,config,target,fk,registered)
    differences=[]
    with np.load(OLD/'R2_SPEED3_PAIR_C0/trace.npz') as z:
        for i in [0,5,10,100,500,1500]:
            q=z['q'][i].copy();t=45.25+float(z['sim_time'][i])
            qo,lo=old.compute(q,0,t,.05);qn,ln=new.compute(q,0,t,.05)
            differences.append(float(abs(qo-qn).max()));old.accept(qo,0);new.accept(qn,0)
    samples=[]
    for phi in [0.,1e-10,1e-8,1e-6,.001,.01,reference.maximum_phase_rad]:
        c=FrontLocalRollingController(reference,config,target,fk,registered);q,l=c.compute(measured,phi,45.25,.05)
        samples.append(dict(phase_rad=phi,target_difference_deg=float(np.degrees(abs(q-target)).max()),
            raw_maximum_deg=float(np.degrees(abs(np.asarray(l['raw_target_rad']))).max()),valid=bool(np.isfinite(q).all())))
    with np.load(OLD/'R2_SPEED3_PAIR_C0/trace.npz') as z:
        old_initial=z['qpos'][0].copy()
    with np.load(BASE/'initial_boundary_pose.npz') as z:initial=z['qpos'].copy()
    initialerror=float(abs(initial-old_initial).max())
    passed=max(differences)<1e-12 and initialerror<1e-12 and all(r['valid'] for r in samples)
    write(BASE/'interface_check.json',dict(physics_steps=0,interface_pass=passed,initial_qpos_vs_old_paired_C0_error=initialerror,
        arbitrary_history_proof='At phi=0 and previous phi=0, compute directly calls R2. All nominal/root/tangent/centers/weights/config/FK/registration equal original; inherited accept commits same target and zero progress. Induction gives equivalence for arbitrary valid encoder/ACK sequences, not just tested first command.',
        observation_target_differences_rad=differences,boundary_samples=samples,
        reuse='R2_SPEED3_PAIR_C0 66.8s; entire zero-phase branch only, not rolling or end hold',
        nonzero_tracking_error='nominal(phi)+retained(previous_ACK)-measured, identical original relation',
        phase_projector='smooth blend old material projector to front phase tangent; source original projector/gain retained',
        coordinate_limitation='encoder cloud transport is conditional shape prior, not world or actual axis measurement',
        C0_perturb_required_before_freeze=True,old_V3_path='NOT_READY'))
    print(json.dumps(dict(interface_pass=passed,maximum_zero_branch_difference_rad=max(differences),initial_error=initialerror)),flush=True)


if __name__=='__main__':main()
