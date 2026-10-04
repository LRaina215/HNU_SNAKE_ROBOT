"""Verify the cleaned controller against both frozen physical-run command logs.

Read-only feedback replay, no physics stepping, actuator access or hashes.
This protects the successful dynamics from accidental changes during cleanup.
"""
import json
import numpy as np

from segmented_pole_experiment import BASE, HORIZONTAL, VERTICAL, pipe_frame, write_json
from mujoco_checkpoint import load_checkpoint
from spatial_takeover_experiment import points_for_angles
from traveling_compliant_winding import TravelingCompliantWinding


def check(name,checkpoint):
    folder=BASE/'traveling_winding'/name
    controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    geometry=json.loads((folder/'geometry.json').read_text(encoding='utf8'))
    z=np.load(folder/'trace.npz')
    m,d,meta,original=load_checkpoint(checkpoint,verify_hashes=False)
    corner,source,target,radius=pipe_frame(m)
    c=TravelingCompliantWinding(geometry[0],original.parameters,d.ctrl,meta['settings']['phase'],
        corner,source,target,radius,measured=np.array(controls[0]['measured']),
        forward_points=lambda q: points_for_angles(m,d,q),pitch_mm=meta['settings']['pitch'])
    errors=[]
    for k,(g,record) in enumerate(zip(geometry,controls)):
        i=k*20
        d.qpos[:]=z['qpos'][i]
        q,_=c.compute(g,np.array(record['measured']),record['phase'],.2,
                       z['normal_force'][max(0,i-59):i+1])
        errors.append(float(abs(q-record['target']).max()))
    assert max(errors)<1e-12,(name,max(errors))
    commands=np.vstack([z['ctrl'][0],*[row['target'] for row in controls]])
    largest=float(np.degrees(abs(np.diff(commands,axis=0)).max()))
    assert largest<=18.+1e-8
    return dict(run=name,commands=len(controls),maximum_reference_error_rad=max(errors),
                maximum_command_step_deg=largest,final_formation=c.progress,
                final_partition=[c.front,c.rear])


if __name__=='__main__':
    result=dict(no_physics_steps=True,no_SHA=True,runs=[
        check('v9_compliant_ends',HORIZONTAL),check('v9_vertical_frozen',VERTICAL)])
    write_json(BASE/'traveling_winding/cleanup_equivalence.json',result)
    print(json.dumps(result,indent=2))
