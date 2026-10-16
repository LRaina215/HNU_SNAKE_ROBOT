"""Nonzero candidate/filter history and clipping attribution, zero integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_1_TRANSMITTED_C_STATE';sys.path.insert(0,str(B))
from direct_controller import DirectController
from physics import MODEL
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(B/'GIVEN_INITIAL.npz');m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);q0=d.qpos.copy();control=DirectController(m,d,r,'DIRECTION',True);control.command(d,0.);control.source_ready=True;control.phase='front_release';control.first_release_t=0.;control.stable_source_P=d.xpos[m.body('head').id:m.body('head').id+21].copy();control.stable_source_R=d.xmat[m.body('head').id:m.body('head').id+21].reshape(21,3,3).copy();control.alpha=.3;control.last_command_transmission_factor=.2;calls=[];original=control.planner.solve
def spy(*a,**kw):
    calls.append(dict(start=control.planner.q.copy(),sent=control.transmitted_geometric.copy()))
    return original(*a,**kw)
control.planner.solve=spy;rows=[]
for t in [.025,.050,.075]:
    previous=control.transmitted_geometric.copy();alpha=control.alpha;fraction=control.last_command_transmission_factor;raw,info=control.command(d,t);candidate=info['geometric_candidate_native_command'];sent=info['geometric_native_command'];expected=previous+info['command_filter_factor']*(candidate-previous);field=info['geometric_curvature_field']+info['transmitted_local_curvature_field'];n=8
    rows.append(dict(t=t,phase_input_fraction=fraction,logged_phase_fraction=info['phase_progress_transmission_fraction'],alpha_increment=control.alpha-alpha,expected_alpha_increment=fraction*.025/r['direct_extension']['release_duration_s'],candidate_filter_factor=info['command_filter_factor'],candidate_delta_deg=float(abs(candidate-previous).max()*180/np.pi),sent_delta_deg=float(abs(sent-previous).max()*180/np.pi),sent_state_identity_error_rad=float(abs(sent-expected).max()),next_planner_sent_identity_error_rad=float(abs(control.planner.q-sent).max()),solve_starts_from_sent_free_C_error_rad=float(abs(calls[-1]['start'][:n]-calls[-1]['sent'][:n]).max()),native_component_sum_error_rad=float(abs(control.planner.coordinates.hinges(field)-raw).max()),given_posterior_total_command_error_rad=float(abs(raw[n:]-initial['ctrl'][n:]).max()),actual_pose_unchanged=bool(np.array_equal(q0,d.qpos))))
limit=np.radians(88);base=np.radians([89.,87.,-89.,-87.]);probe=np.radians([2.,2.,-2.,-2.]);marginal=np.clip(base+probe,-limit,limit)-np.clip(base,-limit,limit);expected=np.radians([0.,1.,0.,-1.]);clipping=dict(base_deg=np.degrees(base).tolist(),requested_probe_deg=np.degrees(probe).tolist(),bounded_marginal_probe_deg=np.degrees(marginal).tolist(),expected_deg=np.degrees(expected).tolist(),error_rad=float(abs(marginal-expected).max()))
passed=all(x['sent_state_identity_error_rad']<1e-12 and x['next_planner_sent_identity_error_rad']<1e-12 and x['solve_starts_from_sent_free_C_error_rad']<1e-12 and x['native_component_sum_error_rad']<1e-12 and x['given_posterior_total_command_error_rad']<1e-12 and x['actual_pose_unchanged'] and abs(x['alpha_increment']-x['expected_alpha_increment'])<1e-12 and abs(x['phase_input_fraction']-x['logged_phase_fraction'])<1e-12 for x in rows) and clipping['error_rad']<1e-12 and any(x['candidate_filter_factor']<.9 and x['candidate_delta_deg']>.01 for x in rows)
out=dict(version=r['version'],simulation_steps=0,passed=passed,rows=rows,clipping=clipping,scope='Three constructed nonzero-error control commands with physical pose held fixed; establishes bookkeeping/phase and native/source command identities only, not physical source hold or path feasibility.')
(B/'TRANSMISSION_HISTORY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
