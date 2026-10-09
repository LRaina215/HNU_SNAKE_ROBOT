"""Copy exact frozen ready-entry trial; default zero-step preparation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,shutil,subprocess
from datetime import datetime,timezone,timedelta
import numpy as np,mujoco
from validation.ready_inner_entry_experiment import BASE
from validation.suspended_supply_experiment import write
from snake_control.suspended_supply_transfer import nodes

def run(args):
    case=BASE/args.case;out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);shutil.copytree(case/'sources',out,dirs_exist_ok=True);shutil.copy2(case/'initial_pose.npz',out/'ready_initial_pose.npz')
    m=mujoco.MjModel.from_xml_path(str(out/'mujoco_model/Snake_Robot_o20_full_source_inner.xml'));d=mujoco.MjData(m);z=np.load(out/'ready_initial_pose.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d);p=nodes(d);c=(p[:-1]+p[1:])/2;r=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1)
    info=dict(status='exact_frozen_sources_ready_without_dynamics',case=args.case,physical_steps=0,source_near_rods=int(np.count_nonzero((r>.075)&(r<.115))),head_mm=p[0]*1000,head_entry_vector_mm=(p[0]-p[1])*1000,initial_state='Same artificial all21 source lower-inner upward start; no initial target hook',output_root=str(out),new_run_requires_new_formal_trial_authorization=True);write(out/'reproduction_ready.json',info);print(json.dumps({k:v for k,v in info.items() if k not in ['head_mm','head_entry_vector_mm']}),flush=True)
    if args.run:
        deadline=args.deadline or (datetime.now(timezone.utc)+timedelta(minutes=20)).isoformat();cmd=[sys.executable,'-u','validation/ready_inner_entry_experiment.py','--name',args.case,'--initial','ready_initial_pose.npz','--deadline',deadline]
        if args.case=='R03':cmd+=['--anchored-feed','--explicit-anchor-roll']
        subprocess.run(cmd,cwd=out,check=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['R01','R02','R03'],default='R03');ap.add_argument('--out',required=True);ap.add_argument('--run',action='store_true',help='Starts a new formal dynamics run; not included in the exhausted R01-R03 allowance');ap.add_argument('--deadline');run(ap.parse_args())
