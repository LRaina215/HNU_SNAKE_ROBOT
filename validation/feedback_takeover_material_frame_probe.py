"""Diagnostic of phase-compensated material-frame odometry from encoders.

Try the rod's rolling phase as a body-orientation predictor, instead of
assuming centre points or shell contact points are stationary. Both signs are
reported to expose convention/model errors; truth scores do not choose one.
"""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context
from feedback_takeover_live_tracking import EncoderPipeTracker


def run():
    root=ROOT/'experiments/segmented_pole_20261005/feedback_takeover';reports=[]
    for name in ('horizontal_encoder_live_whole_coil','vertical_encoder_live_whole_coil'):
        folder=root/name;_,z,m,_,meta=context(folder)
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        plan=json.loads((folder/'plan.json').read_text(encoding='utf8'))
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];times=z['time']-z['time'][0]
        ids=list(range(1,plan['configuration']['initial_front']))
        bodies=[m.body('link'+str(i)).id for i in ids]
        phi=np.interp(times,np.r_[0.,[r['time'] for r in controls]],np.r_[meta['settings']['phase'],[r['front_phase'] for r in controls]])
        q=z['qpos'][0,qadr];load=np.clip(m.actuator_gainprm[:,0]*(z['ctrl'][0]-q),-2.355,2.355)
        tracker=EncoderPipeTracker(m,q,load,meta['settings']['radius'])
        observations=[]
        for i in np.flatnonzero(times<=90.+1e-9)[::2]:
            p=tracker.points(z['qpos'][i,qadr])
            observations.append((i,p,tracker.d.xmat[bodies].reshape(-1,3,3).copy()))
        true_corner,_,true_axis,_=pipe_frame(m)
        for sign in (-1.,1.):
            estimate=dict(tracker.estimate);rows=[]
            old_i,previous,old_rot=observations[0]
            for i,points,rotation in observations[1:]:
                candidates=[]
                for j,module in enumerate(ids):
                    tangent=previous[module+1]-previous[module];tangent/=np.linalg.norm(tangent)
                    local=old_rot[j].T@tangent
                    spin=Rotation.from_rotvec(sign*(phi[i]-phi[old_i])*local).as_matrix()
                    candidates.append(rotation[j]@spin@old_rot[j].T)
                transport=Rotation.from_matrix(candidates).mean().as_matrix()
                old_centers=(previous[:-1]+previous[1:])/2;new_centers=(points[:-1]+points[1:])/2
                shift=new_centers[ids].mean(axis=0)-transport@old_centers[ids].mean(axis=0)
                estimate=dict(estimate,target=transport@estimate['target'],source=transport@estimate['source'],corner_mm=transport@estimate['corner_mm']+shift)
                previous=points;old_rot=rotation;old_i=i
                if i%20:continue
                world=np.zeros(9);mujoco.mju_quat2Mat(world,z['qpos'][i,3:7]);world=world.reshape(3,3)
                rows.append(dict(time=float(times[i]),target_error_deg=float(np.degrees(np.arccos(np.clip((world@estimate['target'])@true_axis,-1,1)))),
                    target_offset_mm=float(np.linalg.norm(np.cross(world@estimate['corner_mm']+1000*z['qpos'][i,:3]-true_corner,true_axis)))))
            reports.append(dict(name=name,phase_sign=sign,rows=rows))
            print(name,sign,'20/40/60/80 axis',[(round(r['time']),round(r['target_error_deg'],2),round(r['target_offset_mm'],2)) for r in rows if min(abs(r['time']-k) for k in (20,40,60,80))<.01],flush=True)
    write_json(root/'material_frame_probe.json',dict(runs=reports,limitation='Commanded material phase is not actual module spin. No CAD correction or closed-loop trial; both conventions reported.'))


if __name__=='__main__':run()
