"""Candidate drape evidence from a fitted probe, without physical truth.

This is a development gate, not a force-closure certificate. The known nominal
robot/contact model and right-angle gravity prior remain explicit assumptions.
It must be validated before automatic approach-to-transfer switching.
"""
import numpy as np
import mujoco
from encoder_pipe_proximity import EncoderPipeProximity
from feedback_takeover_dynamic_observer_probe import virtual_model, predict


def judge(robot, robot_xml, fit, records):
    extension=fit.get('extension')
    model=virtual_model(robot,fit['estimate'],fit['gravity'],robot_xml,extension)
    slots=EncoderPipeProximity(robot).pipe_slots
    source_only=virtual_model(robot,fit['estimate'],fit['gravity'],robot_xml,'source' if extension=='source' else None)
    source_only.geom_contype[slots[1]]=0;source_only.geom_conaffinity[slots[1]]=0
    predicted=predict(source_only,records,fit['initial_velocity'])
    measured=np.array([r['q'] for r in records])
    error=np.degrees(np.asarray(predicted['predicted'])-measured)
    absent_rms=float(np.sqrt(np.mean(error[41:]**2)))
    full_rms=fit['held_out_rms_deg']
    d=mujoco.MjData(model);ids=model.actuator_trnid[:,0]
    qadr=model.jnt_qposadr[ids];vadr=model.jnt_dofadr[ids]
    d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qadr]=records[0]['q']
    d.qvel[:6]=fit['initial_velocity'];d.qvel[vadr]=(measured[1]-measured[0])/.02
    mujoco.mj_forward(model,d);support=[]
    for i,record in enumerate(records):
        force=np.zeros(2)
        for j in range(d.ncon):
            contact=d.contact[j]
            for k,slot in enumerate(slots):
                if slot in (contact.geom1,contact.geom2):
                    wrench=np.empty(6);mujoco.mj_contactForce(model,d,j,wrench)
                    force[k]+=max(0.,wrench[0])
        support.append(force)
        d.ctrl[:]=record['command']
        if i+1<len(records):mujoco.mj_step(model,d,nstep=round(.02/model.opt.timestep))
    fraction=np.mean(np.array(support)[-31:]>.1,axis=0)
    criteria=dict(prediction=full_rms<=.1,
        target_constraint_evidence=absent_rms-full_rms>=.05,
        both_constraints_persist=bool(np.all(fraction>=.9)))
    return dict(passed=all(criteria.values()),criteria=criteria,
        heldout_rms_deg=full_rms,source_only_heldout_rms_deg=absent_rms,
        predicted_support_fraction_last_point6_seconds=fraction,
        predicted_support_N=support,
        limitation='Model-based evidence, not measured contacts or a proof of force closure. Thresholds are engineering development criteria; no hardware/noise qualification.')
