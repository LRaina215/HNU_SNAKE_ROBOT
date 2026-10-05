"""Forward-model diagnostic using only measured joint trajectories/commands.

The model's original environment is erased. Two virtual rays are built from
the encoder-derived initial hypothesis, with the head at identity. Gravity
orientation is an unknown: six scene-compatible candidates are compared by
joint prediction, without using the real floating pose or contact labels.
The six directions assume horizontal/vertical right-angle pipes; this is not
an arbitrary-orientation estimator or a new physical crossing trial.
"""
import copy
import json
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,write_json
from three_segment_review import context
from encoder_pipe_proximity import EncoderPipeProximity


def virtual_model(robot,estimate,gravity,robot_xml,extension=None):
    # Recompile the virtual environment so its static broad-phase tree is
    # rebuilt. Editing only compiled geom poses/sizes leaves stale BVH bounds
    # and may silently remove contacts from this prediction model.
    spec=mujoco.MjSpec.from_file(str(robot_xml));query=EncoderPipeProximity(robot)
    names={'head','tail',*['link'+str(i) for i in range(1,20)]}
    for body in spec.bodies:
        if body.name not in names:
            for geom in body.geoms:geom.contype=0;geom.conaffinity=0
    c=np.asarray(estimate['corner_mm'])/1000
    branches=list(zip(query.pipe_slots,[estimate['source'],estimate['target']],[-1,1]))
    if extension is not None:
        # Additional T arm is a hypothesis, never read from the scene geometry.
        extra=next(i for i in range(robot.ngeom) if i not in query.pipe_slots
            and robot.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER and robot.body_jntnum[robot.geom_bodyid[i]]==0
            and robot.body_parentid[robot.geom_bodyid[i]]==0)
        branches.append((extra,estimate[extension],1 if extension=='source' else -1))
    for slot,axis,sign in branches:
        body=int(robot.geom_bodyid[slot])
        if robot.body_jntnum[body] or robot.body_parentid[body]!=0:raise ValueError('Expected a static world-child storage body')
        parent=spec.find_body(robot.body(body).name);parent.pos[:]=0.;parent.quat[:]=[1.,0.,0.,0.]
        geom=next(g for g in spec.geoms if g.name==robot.geom(slot).name)
        axis=np.asarray(axis)
        radius=estimate['pipe_radius_mm']/1000;half=1.
        geom.fromto[:]=np.r_[c,c+sign*2*half*axis];geom.size[:]=[radius,0.,0.]
        geom.contype=1;geom.conaffinity=1
    spec.option.gravity[:]=gravity;m=spec.compile()
    body_ids=[m.body(name).id for name in names]
    old_ids=[robot.body(name).id for name in names]
    if not np.allclose(m.body_mass[body_ids],robot.body_mass[old_ids],rtol=0,atol=1e-10):
        raise ValueError('Robot XML differs from saved robot masses')
    for field in ('actuator_gainprm','actuator_forcerange','dof_damping'):
        if not np.allclose(getattr(m,field),getattr(robot,field),rtol=0,atol=1e-10):
            raise ValueError('Robot XML differs from saved model: '+field)
    return m


def predict(model,records,initial_velocity=None,initial_joint_velocity=None):
    d=mujoco.MjData(model);ids=model.actuator_trnid[:,0]
    qadr=model.jnt_qposadr[ids];vadr=model.jnt_dofadr[ids]
    d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qadr]=records[0]['q']
    if initial_velocity is not None:d.qvel[:6]=initial_velocity
    d.qvel[vadr]=(np.asarray(records[1]['q'])-records[0]['q'])/.02
    if initial_joint_velocity is not None:d.qvel[vadr]=initial_joint_velocity
    mujoco.mj_forward(model,d);predicted=[];contacts=[]
    for i,record in enumerate(records):
        predicted.append(d.qpos[qadr].copy());contacts.append(d.ncon);d.ctrl[:]=record['command']
        if i+1<len(records):
            mujoco.mj_step(model,d,nstep=round(.02/model.opt.timestep))
    error=np.asarray(predicted)-np.array([r['q'] for r in records])
    return dict(predicted=predicted,predicted_contact_counts=contacts,final_qpos=d.qpos.copy(),final_qvel=d.qvel.copy(),rms_deg=float(np.degrees(np.sqrt(np.mean(error**2)))),
        maximum_error_deg=float(np.degrees(abs(error).max())))


def run():
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials'];reports=[]
    for scene,at in [('horizontal',0.),('vertical',2.)]:
        name=scene+'_encoder_live_whole_coil'
        folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/name
        _,z,m,_,meta=context(folder);time=z['time']-z['time'][0]
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        info=min(controls,key=lambda r:abs(r['time']-.2-at));estimate=info['geometry_observer']['estimate']
        records=next(r['records'] for r in trials if r['scene']==scene and r['snapshot_seconds']==at and r['pattern']=='multisine' and not r['target_removed'])
        source=np.asarray(estimate['source']);target=np.asarray(estimate['target']);normal=np.cross(source,target)
        candidates=[];template=virtual_model(m,estimate,9.81*source,meta['source_model'])
        for label,axis in [('source',source),('target',target),('normal',normal)]:
            for sign in (-1.,1.):
                g=9.81*sign*axis
                private=copy.copy(template);private.opt.gravity[:]=g
                prediction=predict(private,records)
                candidates.append(dict(gravity_label=label,gravity_sign=sign,gravity=g,**prediction))
                print(scene,label,sign,prediction['rms_deg'],prediction['maximum_error_deg'],flush=True)
        chosen=min(candidates,key=lambda r:r['rms_deg'])
        # Read real gravity and initial root orientation only for final scoring.
        i=int(np.argmin(abs(time-at)));rot=np.zeros(9);mujoco.mju_quat2Mat(rot,z['qpos'][i,3:7]);rot=rot.reshape(3,3)
        actual=rot.T@m.opt.gravity
        for candidate in candidates:
            candidate['scoring_gravity_error_deg']=float(np.degrees(np.arccos(np.clip(candidate['gravity']@actual/9.81**2,-1,1))))
        reports.append(dict(scene=scene,time=at,candidates=candidates,chosen_label=chosen['gravity_label'],chosen_sign=chosen['gravity_sign']))
        write_json(root/'dynamic_gravity_probe.json',dict(runs=reports,
            limitation='Diagnostic forward prediction. Virtual geometry from prior encoder estimate; identical nominal robot/contact parameters assumed; unknown initial free velocity set to zero. No physical pose feedback, no controller adoption.'))


if __name__=='__main__':run()
