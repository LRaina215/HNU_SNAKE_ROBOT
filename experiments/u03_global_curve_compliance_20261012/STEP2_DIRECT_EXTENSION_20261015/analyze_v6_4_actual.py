"""Read completed V6.4 phase/load evidence, preserving all older outcomes."""
from pathlib import Path
import json,numpy as np,io
from contextlib import redirect_stdout
P=Path(__file__).resolve().parent;s=(P/'analyze_v6_3_actual.py').read_text(encoding='utf-8').replace('CONTROLLER_V6_3_HEAD_TASK_PROGRESS','CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE').replace('total_actual_starts=37','total_actual_starts=38').replace('V6.3 one actual development run','V6.4 one actual development run')
with redirect_stdout(io.StringIO()):exec(compile(s,str(P/'analyze_v6_3_actual.py'),'exec'),{'__file__':str(P/'analyze_v6_3_actual.py'),'__name__':'__main__'})
C=P/'CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE/T_UPDATE';z=np.load(C/'trace.npz');out=json.loads((C/'OUTCOME_SUMMARY.json').read_text(encoding='utf-8'));release=out['actual_result']['first_release_t'];mask=z['t']>=release if release is not None else np.zeros(len(z['t']),bool);fore=z['eval_pipe_world_N'][:,:2,:8].sum((1,2));out.update(minimum_sampled_fore_source_Fz_N=float(fore[:,2].min()),maximum_sampled_fore_source_abs_Fx_N=float(abs(fore[:,0]).max()),maximum_sampled_fore_source_normal_N=float(z['eval_pipe_normal_N'][:,:2,:8].sum((1,2)).max()),mean_common_fraction_after_release=float(z['command_filter_factor'][mask].mean()) if mask.any() else None,joint18_saturated_sample_duration_s=float(np.trapz((abs(z['torque'][:,17])>=2.3549).astype(float),z['t'])),total_actual_starts=38)
(C/'OUTCOME_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
