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
    mujoco.mj_forward(model,d);support=[];coverage=[];contact_history=[]
    body_to_module={model.body(name).id:i for i,name in enumerate(['head']+['link'+str(i) for i in range(1,20)]+['tail'])}
    target=np.asarray(fit['estimate']['target']);corner=np.asarray(fit['estimate']['corner_mm'])
    h=np.eye(3)[np.argmin(abs(target))];h-=h@target*target;h/=np.linalg.norm(h);v=np.cross(target,h)
    for i,record in enumerate(records):
        force=np.zeros(2);target_modules=set()
        for j in range(d.ncon):
            contact=d.contact[j]
            for k,slot in enumerate(slots):
                if slot in (contact.geom1,contact.geom2):
                    wrench=np.empty(6);mujoco.mj_contactForce(model,d,j,wrench)
                    force[k]+=max(0.,wrench[0])
                    if k==1 and wrench[0]>.1:
                        other=contact.geom2 if contact.geom1==slot else contact.geom1
                        module=body_to_module.get(int(model.geom_bodyid[other]))
                        if module is not None:target_modules.add(module)
        support.append(force)
        points=1000*np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(k)).xanchor for k in range(1,21)],d.site('s_tail').xpos])
        centers=(points[:-1]+points[1:])/2
        active=np.zeros(21,dtype=bool);active[list(target_modules)]=True
        contact_history.append(active)
        # Match the transfer controller's 0.4 s occupancy window. Rolling
        # changes which shells touch; instantaneous simultaneous contact would
        # reject legitimate support as the contact travels around the coil.
        ids=np.flatnonzero(np.mean(contact_history[-20:],axis=0)>.25);span=0.
        if len(ids)>1:
            radial=centers[ids]-corner
            theta=np.sort(np.mod(np.arctan2(radial@v,radial@h),2*np.pi))
            span=2*np.pi-np.max(np.diff(np.r_[theta,theta[0]+2*np.pi]))
        coverage.append(dict(modules=ids,instantaneous_modules=sorted(target_modules),angular_span_deg=float(np.degrees(span))))
        d.ctrl[:]=record['command']
        if i+1<len(records):mujoco.mj_step(model,d,nstep=round(.02/model.opt.timestep))
    fraction=np.mean(np.array(support)[-31:]>.1,axis=0)
    distributed_fraction=float(np.mean([r['angular_span_deg']>90. for r in coverage[-31:]]))
    criteria=dict(prediction=full_rms<=.1,
        target_constraint_evidence=absent_rms-full_rms>=.05,
        both_constraints_persist=bool(np.all(fraction>=.9)),
        target_support_distributed=distributed_fraction>=.9)
    return dict(passed=all(criteria.values()),criteria=criteria,
        heldout_rms_deg=full_rms,source_only_heldout_rms_deg=absent_rms,
        predicted_support_fraction_last_point6_seconds=fraction,
        predicted_target_coverage=coverage,distributed_support_fraction_last_point6_seconds=distributed_fraction,
        predicted_support_N=support,
        limitation='Model-based evidence, not measured contacts or a proof of force closure. Thresholds are engineering development criteria; no hardware/noise qualification.')
