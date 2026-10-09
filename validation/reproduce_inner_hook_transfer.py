"""Reproduce the one inner-hook trial from frozen code and its exact initial pose."""
from pathlib import Path
import sys,argparse,json,shutil,subprocess
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--max-wall',type=float,default=600)
    args=ap.parse_args();case=ROOT/'experiments/unified_curve_transfer_20261007/INNER_HOOK_TRANSFER_20261009/I01';out=args.output_dir.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory: '+str(out))
    shutil.copytree(case/'sources',out);base=out/'experiments/unified_curve_transfer_20261007/INNER_HOOK_TRANSFER_20261009/PREPARE';base.mkdir(parents=True)
    shutil.copy2(case/'initial_pose.npz',base/'initial_pose.npz');g=json.loads((case/'initial_geometry.json').read_text(encoding='utf-8'));(base/'initial_geometry.json').write_text(json.dumps(g['initial_geometry_fit'],indent=2)+'\n',encoding='utf-8')
    old=out/'experiments/unified_curve_transfer_20261007/LOWER_HOOK_TRANSFER_20261009/U04';old.mkdir(parents=True);shutil.copy2(case/'comparison_U04_initial_geometry.json',old/'initial_geometry.json')
    plan=json.loads((case/'plan.json').read_text(encoding='utf-8'));deadline=(datetime.now(timezone.utc)+timedelta(seconds=args.max_wall+300)).isoformat()
    command=[sys.executable,str(out/'validation/inner_hook_transfer_experiment.py'),'--name','I01','--front',str(g['front_modules']),'--bridge',str(g['bridge_modules']),
        '--speed',str(plan['speed_mm_s']/1000),'--rate',str(plan['rate_setting_deg_s']),'--seconds',str(plan['max_actual_seconds']),'--max-wall',str(args.max_wall),'--deadline',deadline]
    entry=dict(original_case=str(case),fresh_workspace=str(out),command=command,frozen_sources=True,exact_saved_artificial_initial_pose=True,active_hook_acquisition_demonstrated=False,additional_dynamics_reproduction_not_performed=True)
    (out/'reproduction_entry.json').write_text(json.dumps(entry,indent=2)+'\n',encoding='utf-8');print(json.dumps(entry,indent=2),flush=True)
    if not args.prepare_only:subprocess.run(command,cwd=out,check=True)


if __name__=='__main__':main()
