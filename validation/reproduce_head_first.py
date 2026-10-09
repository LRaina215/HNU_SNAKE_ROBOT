"""Stage an exact frozen F06/F07 checkout; default is zero-step preparation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,shutil,subprocess
from datetime import datetime,timezone,timedelta
import numpy as np,mujoco
from validation.full_source_inner_experiment import BASE
from validation.suspended_supply_experiment import write
from snake_control.suspended_supply_transfer import nodes

def run(args):
    case=BASE/args.case;out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);shutil.copytree(case/'sources',out,dirs_exist_ok=True)
    rel=BASE.relative_to(ROOT);prep=out/rel/'BRACED_PREPARE_13';prep.mkdir(parents=True);shutil.copy2(case/'initial_pose.npz',prep/'initial_pose.npz');shutil.copy2(case/'initial_geometry.json',prep/'initial_geometry.json')
    pre=out/rel/'HEAD_FIRST_LOWER_PREFLIGHT';pre.mkdir();shutil.copy2(case/'offline_preflight.json',pre/'summary.json')
    m=mujoco.MjModel.from_xml_path(str(out/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'));d=mujoco.MjData(m);z=np.load(prep/'initial_pose.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d);p=nodes(d);c=(p[:-1]+p[1:])/2;r=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1)
    info=dict(status='exact_frozen_sources_ready_without_dynamics',case=args.case,physical_steps=0,source_near_rods=int(np.count_nonzero((r>.075)&(r<.115))),source_radius_mm=r*1000,initial_target_state='all21 rods around source, no preinserted target hook',output_root=str(out),run_requires_new_formal_trial_authorization=True)
    write(out/'reproduction_ready.json',info);print(json.dumps({k:v for k,v in info.items() if k!='source_radius_mm'}),flush=True)
    if args.run:
        deadline=args.deadline or (datetime.now(timezone.utc)+timedelta(minutes=20)).isoformat();cmd=[sys.executable,'-u','validation/head_first_experiment.py','--name',args.case,'--deadline',deadline]
        if args.case=='F07':cmd+=['--actual-exit-gate','--anchored-feed']
        subprocess.run(cmd,cwd=out,check=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['F06','F07'],default='F07');ap.add_argument('--out',required=True);ap.add_argument('--run',action='store_true',help='Runs a new formal physical trial; not included in the exhausted F06/F07 allowance');ap.add_argument('--deadline');run(ap.parse_args())
