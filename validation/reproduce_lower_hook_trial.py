"""Reproduce one bounded lower-hook trial using that trial's frozen sources."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
from datetime import datetime,timedelta,timezone
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from snake_control.hook_pull_review_support import restore_saved_state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['U01','U02','U03','U04'],default='U04')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true',help='build the frozen workspace, without physics')
    ap.add_argument('--max-wall',type=float,default=600)
    args=ap.parse_args();base=ROOT/'experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009'
    case=base/args.case;plan=json.loads((case/'plan.json').read_text(encoding='utf-8'))
    out=args.output_dir.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory: '+str(out))
    shutil.copytree(case/'sources',out)
    initial=out/'experiments/hook_pull_20261006/CLOSE1B';initial.mkdir(parents=True)
    shutil.copy2(case/'initial_saved_state.npz',initial/'final_state.npz')
    # The runner's parent comparison only needs the saved final sample; this
    # is reconstructed from the exact integration state, with zero mj_step.
    m=mujoco.MjModel.from_xml_path(str(out/'mujoco_model/Snake_Robot_o20_lower_hook_transfer.xml'));d=mujoco.MjData(m)
    restore_saved_state(m,d,initial/'final_state.npz')
    np.savez(initial/'trace.npz',qpos=d.qpos[None],qvel=d.qvel[None],ctrl=d.ctrl[None])
    deadline=(datetime.now(timezone.utc)+timedelta(seconds=args.max_wall+300)).isoformat()
    command=[sys.executable,str(out/'validation/hook_start_transfer_experiment.py'),'--name',args.case,
        '--lower-entry','--reform','--posterior-handoff','--speed',str(plan['speed_mm_s']/1000),
        '--rate',str(plan['rate_setting_deg_s']),'--morph-length',str(plan['morph_length_mm']/1000) if 'morph_length_mm' in plan else '.20',
        '--seconds',str(plan['max_actual_seconds']),'--max-wall',str(args.max_wall),'--deadline',deadline]
    if args.case=='U02':command+=['--handoff-exit-count','2']
    if args.case in {'U03','U04'}:command+=['--lower-bridge']
    if args.case=='U04':command+=['--continuous-reform-handoff']
    manifest=dict(original_case=str(case),fresh_workspace=str(out),command=command,
        frozen_sources=True,initial_state_is_artificial=True,parent_comparison_sample_only=True,
        new_actual_run_required=True,reproducibility_is_not_verified_by_an_additional_run=True)
    (out/'reproduction_entry.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
    if not args.prepare_only:subprocess.run(command,cwd=out,check=True)


if __name__=='__main__':main()
