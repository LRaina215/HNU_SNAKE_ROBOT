"""Read actual pressure response and the unchanged preactivation prefix."""
from pathlib import Path
import json,numpy as np,io
from contextlib import redirect_stdout
P=Path(__file__).resolve().parent;s=(P/'analyze_v6_4_actual.py').read_text(encoding='utf-8').replace('CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE','CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF').replace('total_actual_starts=38','total_actual_starts=39').replace('V6.4 one actual development run','V6.5 one actual development run')
with redirect_stdout(io.StringIO()):exec(compile(s,str(P/'analyze_v6_4_actual.py'),'exec'),{'__file__':str(P/'analyze_v6_4_actual.py'),'__name__':'__main__'})
C=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF/T_UPDATE';z=np.load(C/'trace.npz');baseline=np.load(P/'CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE/T_UPDATE/trace.npz');out=json.loads((C/'OUTCOME_SUMMARY.json').read_text(encoding='utf-8'));norm=np.linalg.norm(z['force_curvature_field'],axis=2).max(1)*1000;on=np.flatnonzero(norm>1e-10);first=float(z['t'][on[0]]) if len(on) else None;n=min(len(z['t']),len(baseline['t']));mask=np.abs(z['t'][:n]-baseline['t'][:n])<1e-7
if first is not None:mask&=z['t'][:n]<first-.100001
out.update(maximum_requested_front_relief_per_m=float(norm.max()),first_sampled_requested_relief_t=first,preactivation_prefix_compared_samples=int(mask.sum()),preactivation_prefix_maximum_qpos_difference=float(abs(z['qpos'][:n][mask]-baseline['qpos'][:n][mask]).max()) if mask.any() else None,preactivation_prefix_maximum_ctrl_difference_rad=float(abs(z['ctrl'][:n][mask]-baseline['ctrl'][:n][mask]).max()) if mask.any() else None,prefix_scope='Same sampled times before first nonzero10Hz requested relief, excluding the preceding100ms. Numerical comparison of real saved trajectories, not SHA or a counterfactual rollout.',total_actual_starts=39)
(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
