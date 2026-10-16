"""Meaningful static checks of the requested start and control chain."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from physics import *
import json,pickle,copy,sys
def forbidden(*a,**k):raise RuntimeError('Preparation: mj_step/step1/step2 forbidden')
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,forbidden)
from from_source_curve import FromSourceCurve
from geometry import Geometry,normalize
from source_controller import FromSourceController
from cylinder_estimator import CylinderEstimator
from global_update import GlobalUpdater,probe_command
from source_runner import require_review
from evaluation import snapshot
reg=json.loads((BASE/'REGISTERED.json').read_text());proof=json.loads((BASE/'SOURCE_INITIAL_PROOF.json').read_text())
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(BASE/'SOURCE_INITIAL.npz')
mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
initial_eval=snapshot(m,d);centers=initial_eval['centers'];polar=np.unwrap(np.arctan2(centers[:,2]-1.,centers[:,1]))
geometry_proof=dict(source_winding_span_deg=float(np.ptp(polar)*180/np.pi),adjacent_polar_steps_deg=(np.diff(polar)*180/np.pi).tolist(),
    source_target_self_floor_depth_mm=(initial_eval['depth']*1000).tolist(),source_vertical_N=float(initial_eval['force_world'][0,:,2].sum()),
    initial_actual_joint_max_deg=float(np.max(abs(d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]))*180/np.pi))
checks={};pair={};previews=[];max_raw={};initial_world={}
for condition in reg['conditions']:
    a=FromSourceController(m,d,reg,condition,False);b=FromSourceController(m,d,reg,condition,True);before=d.qpos.copy()
    qa,ia=a.command(d,0.);qb,ib=b.command(d,0.);qa1,ia1=a.command(d,.025);qb1,ib1=b.command(d,.025)
    pair[condition]=dict(initial_pair_raw_equal=bool(np.array_equal(qa,qb)),next_static_pair_raw_equal=bool(np.array_equal(qa1,qb1)),
        initial_ctrl_max_deg=float(np.max(abs(qa-d.ctrl))*180/np.pi),S_at_zero=float(ia['S']),S_after25ms_static=float(ia1['S']),
        source_ready_after25ms=bool(ia1['source_ready']),first_raw_front_contact=a.estimator.first_front_contact,
        controller_did_not_move_qpos=bool(np.array_equal(before,d.qpos)),finite_observed_progress=bool(np.isfinite(ia1['material_progress_observed_m']).all()))
    ref=FromSourceCurve(reg['source_constants'],Geometry.from_dict(reg['conditions'][condition]));ref.configure_initial(d.ctrl)
    points=[];raw_values=[];ss=np.linspace(0,37,75)
    for S in ss:
        raw,info=ref.command(float(S));points.append(info['global_reference_points']);raw_values.append(np.max(abs(raw))*180/np.pi)
    previews.append(np.array(points));max_raw[condition]=float(max(raw_values))
    c=.5*(points[0][:-1]+points[0][1:]);actual=.5*(nodes(d)[:-1]+nodes(d)[1:])
    initial_world[condition]=float(np.sqrt(np.mean(np.sum((c-actual)**2,axis=1)))*1000)
np.savez_compressed(BASE/'STATIC_G_RELEASE_PREVIEW.npz',S=ss,condition_names=np.array(list(reg['conditions'])),G_material_points=np.array(previews),scope='mathematical G path only; not actual dynamics')
# Compare the restored full early family to the actual original implementation.
sys.path.insert(0,str(BASE.parent/'STEP1_20261014'))
with (BASE.parent/'STEP1_20261014/N_SHAPE/controller_checkpoint.pkl').open('rb') as f:old=pickle.load(f)['reference']
for k,v in reg['source_constants'].items():setattr(old,k,np.array(v) if isinstance(v,list) else v)
old.delta=0.;old.source_gamma=0.
equivalence=[]
nominal=FromSourceCurve(reg['source_constants'],Geometry([1.1,.2,1.],[0.,1.,0.],.0565))
for S in [0.,1.,3.,4.5,6.,7.5,9.,15.]:
    qo,io=old.command(S);qn,inn=nominal.command(S)
    equivalence.append(dict(S=S,max_raw_difference_deg=float(np.max(abs(qo-qn))*180/np.pi),max_G_node_difference_m=float(np.max(np.linalg.norm(io['global_reference_points']-inn['global_reference_points'],axis=1)))))
# Raw front contact triggers probing before it can pass any cylinder gate.
prior=Geometry.from_dict(reg['conditions']['DIRECTION']);est=CylinderEstimator(prior,reg['source_known'],reg['estimator'])
sample=dict(material=0,point=np.array([1.5,-.1,.7]),normal=np.array([1.,0.,0.]),normal_N=1.,penetration_m=0.)
events=est.observe(dict(contacts=[sample]),.1,np.zeros(2))
trigger=dict(raw_front_event_t=est.first_front_contact,cylinder_candidate_t=est.first_contact,geometry_reliable=est.last['reliable'],
    first_direction=probe_command(1.1,est.first_front_contact,reg['probe']).tolist(),second_direction=probe_command(5.1,est.first_front_contact,reg['probe']).tolist(),
    synthetic_event_only=True)
# A synthetic PRIOR-based hypothesis exercises write/rollback and theta limits.
class Hypothesis:pass
fake=Hypothesis();fake.geometry=Geometry(prior.entry+[.001,0.,.0005],normalize(prior.axis+[.002,0.,0.]),prior.radius+.0003)
fake.last=dict(reliable=True,t=1.,latest_evidence_t=1.);fake.samples=[]
e,b=prior.frame();phase=FromSourceCurve(reg['source_constants'],prior).receiving_phase(10.)+np.radians(5)
for axial in np.linspace(0,.12,16):
    phi=phase+2*np.pi*axial/reg['source_constants']['pitch']
    point=prior.entry+axial*prior.axis+prior.radius*(np.cos(phi)*e+np.sin(phi)*b)
    fake.samples.append(dict(point=point.tolist()))
update_tests={}
for enabled in [False,True]:
    r=FromSourceCurve(reg['source_constants'],prior);r.configure_initial(d.ctrl);q0,_=r.command(10.)
    u=GlobalUpdater(reg['global_update'],enabled);u.update(r,fake,10.,1.)
    update_tests[str(enabled)]=u.history[-1]
    if not enabled:checks['frozen_proposal_exact_rollback']=bool(np.max(abs(r.command(10.)[0]-q0))<1e-12 and r.theta_offset==0.)
try:require_review(reg);checks['dynamics_entry_blocked']=False
except RuntimeError:checks['dynamics_entry_blocked']=True
checks.update(unique_full_source=proof['single_prescribed_initial'] and proof['all21_source_envelope'],target_and_elbow_no_contact=proof['non_source_pipe_normal_total_N']==0.,
    full_source_winding_over360deg=geometry_proof['source_winding_span_deg']>360.,initial_no_self_or_floor=initial_eval['depth'][2]<=.003 and initial_eval['depth'][3]==0.,initial_pipe_depth_under3mm=max(initial_eval['depth'][:2])<.003,
    source_only_bearing_over90percent=proof['source_straight_vertical_fraction']>=.9,
    initial_raw_matches_ctrl=all(v['initial_ctrl_max_deg']<1e-8 for v in pair.values()),pair_equal_before_write=all(v['initial_pair_raw_equal'] and v['next_static_pair_raw_equal'] for v in pair.values()),
    starts_at_S0_and_holds_for_source_stabilization=all(v['S_at_zero']==0. and v['S_after25ms_static']==0. and not v['source_ready_after25ms'] for v in pair.values()),
    no_qpos_replay=all(v['controller_did_not_move_qpos'] for v in pair.values()),finite_material_observer=all(v['finite_observed_progress'] for v in pair.values()),
    restored_early_family_equivalent=max(v['max_raw_difference_deg'] for v in equivalence)<1e-6 and max(v['max_G_node_difference_m'] for v in equivalence)<1e-8,
    probe_before_cylinder_qualification=est.first_front_contact==.1 and est.first_contact is None and not est.last['reliable'],
    synthetic_update_branch_writes=update_tests['True']['written'],theta_update_exercised=update_tests['True']['theta_after_rad']>0.,
    both_probe_directions_exercised=trigger['first_direction'][0]>.17 and trigger['second_direction'][1]>.17,
    all_proposal_limits_checked='G_curvature_per_m' in update_tests['True']['metrics'])
checks={k:bool(v) for k,v in checks.items()}
out=dict(preparation_only=True,new_dynamics_starts=0,integration_calls=0,checks=checks,initial_geometry_and_contact_proof=geometry_proof,pair_initialization=pair,early_family_equivalence=equivalence,
    raw_front_trigger_test=trigger,synthetic_update_tests=update_tests,static_max_raw_command_deg=max_raw,
    initial_G_vs_actual_same_material_center_rms_mm=initial_world,
    exact_static_equilibrium=False,initial_dynamic_stabilization_not_yet_verified=True,
    static_path_is_not_actual_extension_or_transfer=True,independent_review_passed=False,ready_for_dynamics=False,step2_complete=False)
(BASE/'PRE_DYNAMICS_AUDIT.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(checks=checks,static_max_raw_command_deg=max_raw,initial_world_rms_mm=initial_world,max_original_raw_diff_deg=max(v['max_raw_difference_deg'] for v in equivalence),new_dynamics_starts=0),indent=2))
assert all(checks.values()),'Preparation audit failed; inspect PRE_DYNAMICS_AUDIT.json'
