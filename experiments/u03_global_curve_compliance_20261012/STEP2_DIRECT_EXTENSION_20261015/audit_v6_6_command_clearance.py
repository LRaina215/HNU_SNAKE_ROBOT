"""Compare old/new bounded C solves at three saved actual command states."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,importlib.util,numpy as np,mujoco
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE';old=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF';C=old/'T_UPDATE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
for key in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,key,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No integration')))
spec=importlib.util.spec_from_file_location('old_rough_clearance',old/'rough_planner.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));z=np.load(C/'trace.npz');initial=np.load(B/'GIVEN_INITIAL.npz');records=[json.loads(line) for line in (C/'contact_evidence.jsonl').open(encoding='utf-8')];rows=[]
for target in [2.1,22.3,25.]:
    i=int(np.argmin(abs(z['t']-target)));t=float(z['t'][i]);factor=float(z['command_filter_factor'][i]);rec=min(records,key=lambda row:abs(row['t']-t));contacts=rec['contacts']
    for ct in contacts:
        for key in ['point','normal','world_force_N']:ct[key]=np.asarray(ct[key])
    outcomes=[]
    for name,cls in [('old_geometric_only',legacy.RoughPlanner),('new_both_charts',RoughPlanner)]:
        plan=cls(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);plan.observe(z['qpos'][i],int(z['stance_anchor_material'][i]));plan.observe_contacts(contacts);qphysical=plan.snapshot['qpos'].copy();n=14 if z['beta'][i]>0 else 8
        local=sum(z[key][i] for key in ['probe_curvature_field','force_curvature_field','preload_curvature_field','receiving_compliance_curvature_field','source_force_curvature_field']);requested=np.clip(z['geometric_candidate_native_command'][i]+plan.coordinates.hinges(local),-np.radians(88),np.radians(88));sent=z['raw_native_command'][i];previous_total=(sent-factor*requested)/(1-factor);previous=z['previous_transmitted_geometric_native_command'][i].copy();previous[n:]=z['qpos'][i,plan.qa[n:]];plan.q=previous.copy();offset=previous_total-previous;plan.command_local_offset=offset.copy()
        G=z['global_reference_points'][i].copy();Ctarget=z['geometric_reference_points'][i].copy();tangent=Ctarget[0]-Ctarget[1];tangent/=np.linalg.norm(tangent)
        # Isolate feasibility using the same saved controller-generated goals.
        # No target truth or new runtime path is read by the controller.
        plan.guide=lambda *a:(G.copy(),Ctarget[0].copy(),tangent.copy(),n)
        plan.wrap_guide=lambda *a:(Ctarget.copy(),Ctarget[0].copy(),tangent.copy(),n)
        candidate,_,metrics=plan.solve(float(z['alpha'][i]),float(z['beta'][i]),0.,position_cap=.002,joint_step_rad=np.radians(20)*.025)
        command=candidate+offset;delta=plan.coordinates.field(command-previous_total);peak=np.linalg.norm(delta,axis=1).max()*1000;rate=min(1.,1.5*.025/max(peak,1e-12),np.radians(20)*.025/max(abs(command-previous_total).max(),1e-12));before=plan.command_step_metrics(previous_total,previous_total+rate*(command-previous_total));f=rate
        for _ in range(16):
            step=plan.command_step_metrics(previous_total,previous_total+f*(command-previous_total))
            if step['maximum_geometry_step_m']<=.002+1e-10 and max(step['measured_plane_excess_compression_m'],step['known_source_command_excess_compression_m'])<=2e-6:break
            f*=.5
        else:f=0.
        outcomes.append(dict(policy=name,solver=metrics,initial_rate_fraction=rate,common_fraction=f,unshortened_metrics=before,sent_step_metrics=step,command_step_joint_deg=float(abs(command-previous_total).max()*180/np.pi),offset_peak_deg=float(abs(offset).max()*180/np.pi),snapshot_unchanged=bool(np.array_equal(qphysical,plan.snapshot['qpos']))))
    rows.append(dict(saved_t=t,goals='Same saved controller G/C targets for both isolated solves',results=outcomes))
out=dict(version=r['version'],simulation_steps=0,rows=rows,scope='Three saved physical poses/contacts and controller targets. One original-bounded private solve per policy/state; previous local offset held fixed. This tests command feasibility, not time integration, actual load response, path completion or source retention.')
(B/'COMMAND_CLEARANCE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
