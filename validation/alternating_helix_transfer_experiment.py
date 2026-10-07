"""Actual dynamics runner for current full-transfer task, separate ledger."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import argparse,json,time,shutil,csv
import numpy as np
import mujoco
from hook_pull_experiment import Sim,replay_video
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.alternating_helix_transfer import AlternatingHelixTransfer,DT
BASE=ROOT/'experiments/alternating_helix_transfer_20261007'
INITIAL=ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz'
def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf8')
def run(name,seconds,variant,note,initial=INITIAL):
    case=BASE/name;case.mkdir(parents=True,exist_ok=False)
    for source in ['snake_control/alternating_helix_transfer.py','validation/alternating_helix_transfer_experiment.py']:
        target=case/'sources'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    write(case/'plan.json',dict(change=note,variant=variant,seconds=seconds,start=str(initial),physics='unmodified original model; joint ctrl only',feedback='actual simulated pose, encoder and contact observations; not hardware-ready'))
    sim=Sim(case);restore_saved_state(sim.m,sim.d,initial)
    t0=sim.d.time;sim.stage='start';sim.psi=0.;sim.snapshot()
    controller=AlternatingHelixTransfer(sim.m,sim.d,variant)
    outcome='duration_limit_incomplete';start=time.monotonic();lost=0;complete_ticks=0
    try:
        for tick in range(round(seconds/DT)):
            target,info=controller.command(sim.d,tick*DT,sim.rows[-1]['contact_force_N'])
            sim.stage=info['stage'];sim.commands.append(dict(time_s=tick*DT,target_rad=target.tolist(),**info))
            sim.step_command(target)
            r=sim.rows[-1];c=r['centers_mm'];force=r['contact_force_N']
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            if r['ground']:outcome='ground_contact';break
            if np.max(abs(sim.d.qpos[sim.qadr]))>np.radians(90.5):outcome='actual_joint_limit';break
            lost=lost+1 if force[1].sum()<.1 else 0
            if lost>=40:outcome='target_contact_lost_2s';break
            on_target=(c[:,1]>80).all() and (np.linalg.norm(c[:,[0,2]]-[1100,1000],axis=1)<130).all()
            supported=on_target and np.count_nonzero(force[1]>.1)>=4 and force[0].sum()<.1 and not r['self_contact']
            complete_ticks=complete_ticks+1 if supported else 0
            if complete_ticks>=40:
                outcome='complete_transfer';break
            if tick%100==99:
                print(json.dumps(dict(t_s=round((tick+1)*DT,2),stage=sim.stage,front_y_mm=round(float(c[:8,1].mean()-sim.rows[0]['centers_mm'][:8,1].mean()),2),rear_x_mm=round(float(c[13:,0].mean()-sim.rows[0]['centers_mm'][13:,0].mean()),2),target=np.flatnonzero(force[1]>.1).tolist(),source=np.flatnonzero(force[0]>.1).tolist(),max_q_deg=float(np.degrees(abs(r['q'])).max()))),flush=True)
    except Exception as e:
        outcome='exception';write(case/'error.json',dict(error=repr(e)))
        import traceback;traceback.print_exc()
    finally:sim.save()
    c0=sim.rows[0]['centers_mm'];c1=sim.rows[-1]['centers_mm']
    result=dict(name=name,outcome=outcome,physical_seconds=sim.d.time-t0,wall_seconds=time.monotonic()-start,front_mean_y_advance_mm=float((c1[:8]-c0[:8])[:,1].mean()),rear_mean_x_advance_mm=float((c1[13:]-c0[13:])[:,0].mean()),tail_center_mm=c1[-1],final_centers_mm=c1,final_target_modules=np.flatnonzero(sim.rows[-1]['contact_force_N'][1]>.1),final_source_modules=np.flatnonzero(sim.rows[-1]['contact_force_N'][0]>.1),max_command_deg=float(np.degrees(abs(np.array([r['ctrl'] for r in sim.rows]))).max()),max_joint_deg=float(np.degrees(abs(np.array([r['q'] for r in sim.rows]))).max()),max_torque_Nm=float(abs(np.array([r['torque'] for r in sim.rows])).max()),events=controller.events,complete=outcome=='complete_transfer',external_forces_zero=bool(not np.any(sim.d.xfrc_applied) and not np.any(sim.d.qfrc_applied)))
    write(case/'result.json',result)
    with (BASE/'trial_ledger.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(name=name,change=note,outcome=outcome,seconds=result['physical_seconds']),ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),flush=True)
    # Replay actual qpos from every 50 ms, preserving the whole physical timeline.
    replay_video(case,[dict(r,sim_time=r['sim_time']-t0) for r in sim.rows])
    import cv2
    video=cv2.VideoCapture(str(case/'replay_1x.mp4'))
    write(case/'video.json',dict(fps=video.get(cv2.CAP_PROP_FPS),frames=video.get(cv2.CAP_PROP_FRAME_COUNT),physical_seconds=result['physical_seconds'],speed=1.,source='saved physical qpos, no predicted root poses'))
    video.release()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--initial',type=Path,default=INITIAL);a.add_argument('--name',required=True);a.add_argument('--seconds',type=float,default=60);a.add_argument('--variant',type=int,choices=range(1,12),default=9);a.add_argument('--note',required=True);args=a.parse_args();run(args.name,args.seconds,args.variant,args.note,args.initial)
