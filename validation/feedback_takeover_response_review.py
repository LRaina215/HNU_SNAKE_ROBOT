"""Post-run truth scoring of response-based control; never a feedback source."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import pipe_frame, write_json


def review(folder):
    _, z, m, d, _ = context(folder)
    controls = json.loads((folder/'controls.json').read_text(encoding='utf8'))
    plan = json.loads((folder/'plan.json').read_text(encoding='utf8'))
    metrics = json.loads((folder/'process_metrics.json').read_text(encoding='utf8'))
    with z:
        trace = {k: z[k] for k in ('time','qpos','qvel','points_mm','torque','normal_force')}
    t = trace['time']-trace['time'][0]
    corner, _, axis, _ = pipe_frame(m)
    rows = []
    for control in controls:
        at = control['time']-plan['arguments']['dt']
        i = int(np.argmin(abs(t-at)))
        q = trace['qpos'][i]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q[3:7]); rotation = rotation.reshape(3,3)
        observer = control['geometry_observer']; est = observer['estimate']
        line = rotation @ est['target']
        point = rotation @ est['corner_mm']+1000*q[:3]
        rows.append(dict(time=float(t[i]),
            target_error_deg=float(np.degrees(np.arccos(np.clip(line@axis,-1,1)))),
            target_offset_mm=float(np.linalg.norm(np.cross(point-corner,axis))),
            innovation_rms_deg=observer.get('innovation_rms_deg'),
            topology=observer.get('topology')))
    centers = (trace['points_mm'][:,:-1]+trace['points_mm'][:,1:])/2000
    velocity = np.diff(centers,axis=0)/np.diff(t)[:,None,None]
    speed = np.linalg.norm(velocity,axis=2)
    valid = (t[1:]>=1)&(t[1:]<=metrics['metrics_end_seconds']+1e-6)
    i, module = np.unravel_index(np.where(valid[:,None],speed,-1).argmax(),speed.shape)
    i += 1
    ci = min(len(controls)-1,int(np.searchsorted([c['time'] for c in controls],t[i])))
    control = controls[ci]
    sub = min(len(control['delivery_feedback'])-1,max(0,int((t[i]-control['time']+plan['arguments']['dt'])/plan['arguments']['delivery_period'])))
    delivered = control['delivery_feedback'][sub]
    reference = plan['configuration']['initial_front']
    d.qpos[:] = trace['qpos'][i]; d.qvel[:] = trace['qvel'][i]
    mujoco.mj_forward(m,d)
    jac = np.zeros((3,m.nv)); rot_jac = np.zeros_like(jac)
    mujoco.mj_jac(m,d,jac,rot_jac,centers[i,reference],m.body('link'+str(reference)).id)
    rigid = jac@d.qvel+np.cross(rot_jac@d.qvel,centers[i,module]-centers[i,reference])
    world = velocity[i-1,module]
    peak = dict(time=float(t[i]),module=int(module),world_speed=float(np.linalg.norm(world)),
        world_velocity=world,reference_module=reference,reference_rigid_component=rigid,
        relative_speed=float(np.linalg.norm(world-rigid)),relative_component=world-rigid,
        filtered_relative_speed=delivered['body_relative_speed'],
        braking_torque=delivered['body_damping_torque'],actuator_torque=trace['torque'][i],
        front=control['front'],rear=control['rear'],actual_prefix=control['actual_prefix'],
        wrap=control['front_wrap'],grip=control['grip'],
        source_contact_modules=np.flatnonzero(trace['normal_force'][i,0]>.1),
        target_contact_modules=np.flatnonzero(trace['normal_force'][i,1]>.1))
    errors=np.array([r['target_error_deg'] for r in rows])
    report=dict(run=folder.name,rows=rows,peak_velocity=peak,
        target_axis_error_deg=dict(median=float(np.median(errors)),p95=float(np.quantile(errors,.95)),maximum=float(errors.max())),
        limitation='Post-run physical truth is used only to score the saved estimates and decompose the speed peak. It never corrects the controller.')
    write_json(folder/'response_review.json',report)
    print(json.dumps(dict(run=folder.name,axis=report['target_axis_error_deg'],peak={k:peak[k] for k in ('time','module','world_speed','relative_speed','filtered_relative_speed','front','rear','actual_prefix','wrap')}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    review(p.parse_args().folder)
