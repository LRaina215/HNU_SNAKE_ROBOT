"""Independent contact/pose audit of the oracle-motion training trials."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
from contact_turn_experiment import BASE,write_json,sha


def audit(case):
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    manifest=json.loads((case/'manifest.json').read_text(encoding='utf8'))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    d=mujoco.MjData(m)
    assert m.nu==20 and m.opt.timestep==.001 and np.all(m.actuator_gainprm[:,0]==127)
    assert np.array_equal(m.actuator_forcerange,np.tile([-2.355,2.355],(20,1)))
    assert manifest['model_hash']==sha(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml')
    head=m.body('head').id;floor=m.geom('floor').id
    target={m.geom(name).id for name in ('pipe_sharp_out_03','pipe_straight_04')}
    with np.load(case/'trace.npz') as r:
        t=r['sim_time']
        assert np.allclose(np.diff(t),.01,rtol=0,atol=1e-10)
        assert np.max(abs(np.clip(127*(r['ctrl']-r['q']),-2.355,2.355)-r['torque']))<1e-10
        controls=json.loads((case/'control.json').read_text(encoding='utf8'))
        max_contacts=0;ground=False;self_contact=False;initial_ncon=None;max_fk_error=0.
        counts=[]
        # Audit every decision state. The raw 100 Hz trajectory remains intact;
        # this independently checks contact ownership at actual control times.
        indices=[0]+[int(np.argmin(abs(t-c['time_s']))) for c in controls]
        for i in indices:
            d.qpos[:]=r['qpos'][i];d.qvel[:]=r['qvel'][i];d.ctrl[:]=r['ctrl'][i]
            mujoco.mj_forward(m,d)
            if i==0:initial_ncon=int(d.ncon)
            modules=set()
            for c in d.contact:
                if c.geom1==floor or c.geom2==floor:ground=True
                b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                self_contact |= bool(b1>=head and b2>=head and abs(b1-b2)>1)
                if c.geom1 in target and b2>=head and c.pos[1]>.025:modules.add(b2-head)
                if c.geom2 in target and b1>=head and c.pos[1]>.025:modules.add(b1-head)
            counts.append(len(modules));max_contacts=max(max_contacts,len(modules))
            pts=np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{j}').xanchor for j in range(1,21)],d.site('s_tail').xpos])
            max_fk_error=max(max_fk_error,float(abs(pts-r['points'][i]).max()))
        assert initial_ncon==0 and max_fk_error<1e-12
        assert max_contacts==result['maximum_target_contact_modules']
        assert counts[1:]==[len(c['target_contact_modules']) for c in controls]
        commands=json.loads((case/'commands.json').read_text(encoding='utf8'))
        assert all(x['applied'] and x['computed_sim_time']==x['applied_sim_time'] for x in commands)
        assert [x['command_id'] for x in commands]==list(range(1,len(commands)+1))
        assert np.degrees(abs(r['ctrl'])).max()<=90
        motion=r['stage']=='motion'
        value=dict(label=case.name,outcome=result['outcome'],success=result['success'],
                   decision_states_audited=len(controls),raw_samples=len(t),max_fk_error_m=max_fk_error,
                   actual_target_contact_modules=max_contacts,actual_target_contact_at_start=counts[0],
                   ground_contact=ground,self_contact=self_contact,
                   maximum_head_target_progress_m=float(r['points'][motion,0,1].max()),
                   final_head_target_progress_m=float(r['points'][-1,0,1]),
                   final_head_world_m=r['points'][-1,0].tolist(),
                   tracking_max_deg=float(np.degrees(abs(r['ctrl'][motion]-r['q'][motion])).max()),
                   torque_rms_nm=float(np.sqrt(np.mean(r['torque'][motion]**2))),
                   saturation_fraction=float(np.mean(abs(r['torque'][motion])>=2.355*.999)),
                   trace_sha256=sha(case/'trace.npz'),command_sha256=sha(case/'commands.json'),
                   source_hashes=manifest['code_hash'])
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path);args=p.parse_args()
    if args.case:
        row=audit(args.case);write_json(args.case/'independent_audit.json',row);print(json.dumps(row));return
    raw=json.loads((BASE/'stage1_training_results.json').read_text(encoding='utf8'))
    assert raw['completed']==12
    rows=[]
    for item in raw['results']:
        row=audit(ROOT/item['directory']);rows.append(row)
        write_json(BASE/'stage1_training_independent_audit_progress.json',dict(completed=len(rows),required=12,rows=rows))
        print(json.dumps({k:row[k] for k in ('label','outcome','actual_target_contact_modules','tracking_max_deg')}),flush=True)
    best=max(rows,key=lambda x:(x['success'],x['actual_target_contact_modules'],
             next(y['max_target_wrap_rad'] for y in raw['results'] if y['label']==x['label']),x['maximum_head_target_progress_m']))
    result=next(x for x in raw['results'] if x['label']==best['label'])
    selected_manifest=json.loads((ROOT/result['directory']/'manifest.json').read_text(encoding='utf8'))
    freeze=dict(stage='known_direction_motion',training_completed=12,training_successes=sum(x['success'] for x in rows),
                selected_label=best['label'],front=selected_manifest['front'],width_mm=selected_manifest['width_mm'],
                turn_deg=selected_manifest['maximum_turn_deg'],test_states=json.loads((BASE/'stage1_training_plan.json').read_text(encoding='utf8'))['test_states'],
                audit_sha256=sha(__file__),training_plan_sha256=sha(BASE/'stage1_training_plan.json'),
                training_results_sha256=sha(BASE/'stage1_training_results.json'),selected_manifest=selected_manifest,
                note='Selection for required five-state verification; this does not release stage 2.')
    write_json(BASE/'stage1_training_independent_audit.json',dict(passed=True,rows=rows))
    write_json(BASE/'stage1_parameter_freeze.json',freeze)


if __name__=='__main__':main()
