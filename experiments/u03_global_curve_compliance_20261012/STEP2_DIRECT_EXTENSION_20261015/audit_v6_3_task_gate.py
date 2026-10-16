"""Construct a rejected C solve to check bounded phase lead without integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_3_HEAD_TASK_PROGRESS';sys.path.insert(0,str(B))
from direct_controller import DirectController
from physics import MODEL
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(B/'GIVEN_INITIAL.npz');rows=[]
for alpha in [.02,.3]:
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);mujoco.mj_setState(m,d,initial['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);q0=d.qpos.copy();control=DirectController(m,d,r,'DIRECTION',True);control.command(d,0.);control.source_ready=True;control.first_release_t=0.;control.phase='front_release';control.stable_source_P=d.xpos[m.body('head').id:m.body('head').id+21].copy();control.stable_source_R=d.xmat[m.body('head').id:m.body('head').id+21].reshape(21,3,3).copy();control.alpha=alpha
    def rejected(a,b=0.,theta=0.,**kwargs):
        G,target,T,n=control.planner.guide(a,b,theta);control.planner.last_geometric_targets=G.copy();control.planner.last_guide=G.copy();metric=dict(accepted=False,solution_kind='hold',initial_clearance_violation_m=.0008,head_error_m=float(np.linalg.norm(control.planner.fk(control.planner.q)[0]-target)));return control.planner.q.copy(),G,metric
    control.planner.solve=rejected;first,info=control.command(d,.025);before=control.alpha;previous_ready=control.geometric_progress_ready;fraction=control.last_command_transmission_factor;raw,next_info=control.command(d,.050);after=control.alpha
    rows.append(dict(constructed_alpha=alpha,solver_stub='Rejected, held C, initial clearance0.8mm; no claim of a real solver/physical trajectory',head_sent_error_mm=info['sent_geometric_head_tracking_error_m']*1000,head_actual_error_mm=info['actual_head_task_error_m']*1000,ready_next=previous_ready,alpha_before_second=before,alpha_after_second=after,expected_second_increment=fraction*.025/25 if previous_ready else 0.,actual_second_increment=after-before,actual_pose_unchanged=bool(np.array_equal(q0,d.qpos)),source_given_command_error_rad=float(abs(raw[8:]-initial['ctrl'][8:]).max()),native_integral_error_rad=float(abs(control.planner.coordinates.hinges(next_info['curvature_field'])-raw).max())))
passed=rows[0]['ready_next'] and rows[0]['actual_second_increment']>0 and not rows[1]['ready_next'] and rows[1]['actual_second_increment']==0 and all(abs(row['actual_second_increment']-row['expected_second_increment'])<1e-12 and row['actual_pose_unchanged'] and row['source_given_command_error_rad']<1e-12 and row['native_integral_error_rad']<1e-12 for row in rows)
out=dict(version=r['version'],simulation_steps=0,passed=passed,rows=rows,rough_planner_byte_identical_to_reviewed_V6_2=(B/'rough_planner.py').read_bytes()==(P/'CONTROLLER_V6_2_FEASIBILITY_RESTORE/rough_planner.py').read_bytes(),scope='Constructed solver rejection checks only phase policy and native/source identities, not path feasibility, source retention, release success or contact. Actual pose is held fixed and integration functions raise.')
(B/'HEAD_TASK_PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
