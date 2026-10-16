"""Restore an infeasible sent C within unchanged bounds, gate phase by sent task."""
from pathlib import Path
import shutil,json
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V6_1_TRANSMITTED_C_STATE';B=P/'CONTROLLER_V6_2_FEASIBILITY_RESTORE';B.mkdir(exist_ok=False)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
r=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));r.update(version='DIRECT_EXTENSION_V6_2_FEASIBILITY_RESTORE',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,new_dynamics_starts=0,prior_dynamics_starts_preserved=27,prior_legacy_postwrap_starts_preserved=8,prior_version='V6.1 failed32s before contact: source drift12.005269mm, C stalled after~4s while alpha reached1',revision_reason='Numerical feasibility restoration for already-infeasible sent C; prevent alpha/beta advancing on a rejected or far-behind geometric task. No physical or clearance limit relaxation.')
r['focused_corrections']=dict(baseline='V6.1 transmitted C state, bounded marginal probes, V6 load reduction and original given grip',restoration='Only when normal tracking solve is rejected and initial clearance violation exceeds original50um acceptance: bounded feasibility objective with original hard2mm motion constraints, then original nonlinear clearance/motion acceptance. May increase tracking objective only from an infeasible start.',phase='Existing tracking/fraction gating additionally requires previous geometric acceptance and sent-C head target error within existing20mm rough receiver uncertainty',unchanged='Original initial/physics/G inputs, original source12mm/12degree/penetration/self/floor/torque bounds, exact native material grid, all probe and estimator requirements')
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(B/'register_direct.py').write_text('"""Prepared by ../prepare_v6_2_feasibility_restore.py; never reset a frozen case."""\n',encoding='utf-8')
f=B/'rough_planner.py';s=f.read_text(encoding='utf-8')
a="        evaluate(previous[:n]);initial_objective=float(.5*cache['r']@cache['r']);step=candidate-previous;accepted=False;backtracks=0"
b="        evaluate(previous[:n]);initial_objective=float(.5*cache['r']@cache['r']);initial_violation=float(np.max(cache['limits']-cache['gaps']));step=candidate-previous;accepted=False;backtracks=0;restoration=None;solution_kind='hold'"
assert s.count(a)==1;s=s.replace(a,b)
a="        if accepted:self.q=candidate\n        else:evaluate(previous[:n])"
b="""        if accepted:solution_kind='tracking'
        if not accepted and initial_violation>.00005:
            # Restore an already-infeasible sent C; retain original joint and
            # nonlinear motion/clearance acceptance rather than accepting a jump.
            def feasibility(u):
                con(u);v=np.minimum(0.,(cache['gaps']-cache['limits'])/.001);return .5*v@v
            def feasibility_gradient(u):
                con(u);v=np.minimum(0.,(cache['gaps']-cache['limits'])/.001);return scale*cache['gradients'].T@v/.001
            motion_constraints=[] if position_cap is None else [dict(type='ineq',fun=lambda u:(evaluate(previous[:n]+scale*u),cache['motion'])[1],jac=lambda u:(evaluate(previous[:n]+scale*u),scale*cache['motion_J'])[1])]
            restoration=minimize(feasibility,np.zeros(n),jac=feasibility_gradient,bounds=list(zip(lo,hi)),method='SLSQP',constraints=motion_constraints,options=dict(maxiter=max_iterations or self.p['max_iterations'],ftol=1e-8,disp=False))
            step=np.zeros(20);step[:n]=scale*restoration.x
            for backtracks in range(13):
                candidate=previous+step*(.5**backtracks);evaluate(candidate[:n]);violation=float(np.max(cache['limits']-cache['gaps']))
                accepted=bool(np.all(np.isfinite(candidate)) and violation<=.00005 and (position_cap is None or cache['motion'].min()>=-1e-7))
                if accepted:solution_kind='feasibility_restoration';break
        if accepted:self.q=candidate
        else:evaluate(previous[:n])"""
assert s.count(a)==1;s=s.replace(a,b)
a="self.last_guide=G.copy();self.last_metrics=dict(accepted=accepted,solver_success=bool(fit.success),solver_iterations=int(fit.nit),clearance_violation_m=violation,"
b="self.last_guide=G.copy();self.last_metrics=dict(accepted=accepted,solution_kind=solution_kind,initial_clearance_violation_m=initial_violation,restoration_attempted=restoration is not None,restoration_solver_success=bool(restoration.success) if restoration is not None else False,restoration_iterations=int(restoration.nit) if restoration is not None else 0,solver_success=bool(fit.success),solver_iterations=int(fit.nit),clearance_violation_m=violation,"
assert s.count(a)==1;s=s.replace(a,b);f.write_text(s,encoding='utf-8')
f=B/'direct_controller.py';s=f.read_text(encoding='utf-8');s=s.replace('self.last_command_transmission_factor=1.','self.last_command_transmission_factor=1.;self.geometric_progress_ready=True;self.sent_geometric_head_tracking_error_m=0.',1)
s=s.replace('phase_progress_fraction=self.last_command_transmission_factor;sensor=', 'phase_progress_fraction=self.last_command_transmission_factor;phase_geometric_ready=self.geometric_progress_ready;sensor=',1)
a="tracking=bool(max(abs(sensor['joint_actual']-self.raw).max(),abs(self.raw-self.last_geo).max())<=np.radians(self.p['progress_tracking_gate_deg']))"
b=a+' and phase_geometric_ready';assert s.count(a)==1;s=s.replace(a,b)
a='self.planner.q=self.transmitted_geometric.copy();sent_geometric_field=coordinates.field(self.transmitted_geometric)'
b=a+"\n        if self.source_ready and self.post is None:\n            sent_C=self.planner.fk(self.transmitted_geometric);self.sent_geometric_head_tracking_error_m=float(np.linalg.norm(sent_C[0]-self.planner.last_geometric_targets[0]));self.geometric_progress_ready=bool(metrics.get('accepted',False) and self.sent_geometric_head_tracking_error_m<=self.reg['kinematic_solver']['rough_receiver_uncertainty_m'])"
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('phase_progress_transmission_fraction=phase_progress_fraction,probe_curvature_per_m=', 'phase_progress_transmission_fraction=phase_progress_fraction,phase_geometric_progress_ready=phase_geometric_ready,geometric_progress_ready_next_command=self.geometric_progress_ready,sent_geometric_head_tracking_error_m=self.sent_geometric_head_tracking_error_m,probe_curvature_per_m=',1);f.write_text(s,encoding='utf-8');print(str(B))
