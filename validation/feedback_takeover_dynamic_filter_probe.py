"""Longer recorded-response test of the private dynamics/encoder EKF."""
import argparse
import json
import time
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from feedback_takeover_dynamic_observer_probe import virtual_model
from feedback_takeover_dynamic_filter import DynamicsFilter,TopologyDynamicsFilter


def run(scene,stop,flow,velocity_observation,bounded,covariance_bound,aligned_gravity,topology):
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    suffix='_aligned_gravity' if aligned_gravity else '_joint_velocity_precise_central' if scene=='vertical' else ''
    fit=json.loads((root/('dynamic_fit_'+scene+suffix+'.json')).read_text(encoding='utf8'))
    folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/(scene+'_encoder_live_whole_coil')
    _,z,robot,_,meta=context(folder)
    recorded_qpos=z['qpos'];recorded_ctrl=z['ctrl']
    model=virtual_model(robot,fit['estimate'],fit['gravity'],meta['source_model'])
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    times=z['time']-z['time'][0];start=int(np.argmin(abs(times-fit['time'])))
    velocity=np.asarray(fit['parameters'][10:16]) if fit.get('fit_velocity',False) else None
    joint_velocity=(recorded_qpos[start+1,qadr]-recorded_qpos[start,qadr])/.01
    models={'none':model}
    if topology:
        for extension in ('source','target'):models[extension]=virtual_model(robot,fit['estimate'],fit['gravity'],meta['source_model'],extension)
    cls=TopologyDynamicsFilter if topology else DynamicsFilter
    observer=cls(models if topology else model,recorded_qpos[start,qadr],velocity,joint_velocity,flow_linearization=flow,velocity_observation=velocity_observation,bounded_update=bounded,covariance_bound=covariance_bound)
    rows=[];begin=time.perf_counter();failure=None;latencies=[]
    corner,_,target,_=pipe_frame(robot)
    for i in range(start+2,min(len(times),int(round(stop/.01))+1),2):
        tick=time.perf_counter();info=observer.step(recorded_ctrl[i],recorded_qpos[i,qadr],.02);latencies.append(time.perf_counter()-tick);d=observer.data
        if not np.all(np.isfinite(d.qpos)) or np.linalg.norm(d.qpos[:3])>3.:
            failure=dict(time=float(times[i]),reason='Private state diverged beyond 3 m or became nonfinite');break
        if i%20:continue
        estimated_rot=np.zeros(9);mujoco.mju_quat2Mat(estimated_rot,d.qpos[3:7]);estimated_rot=estimated_rot.reshape(3,3)
        local_target=estimated_rot.T@np.asarray(fit['estimate']['target'])
        local_corner=estimated_rot.T@(np.asarray(fit['estimate']['corner_mm'])-1000*d.qpos[:3])
        actual_rot=np.zeros(9);mujoco.mju_quat2Mat(actual_rot,recorded_qpos[i,3:7]);actual_rot=actual_rot.reshape(3,3)
        error=float(np.degrees(np.arccos(np.clip((actual_rot@local_target)@target,-1,1))))
        offset=float(np.linalg.norm(np.cross(actual_rot@local_corner+1000*recorded_qpos[i,:3]-corner,target)))
        rows.append(dict(time=float(times[i]),target_error_deg=error,target_offset_mm=offset,
            private_contacts=d.ncon,estimated_qpos=d.qpos.copy(),estimated_qvel=d.qvel.copy(),**info))
        if i%500==0:print(scene,round(times[i],2),'axis',round(error,3),'offset',round(offset,3),'innovation',info['innovation_rms_deg'],'wall',round(time.perf_counter()-begin,1),flush=True)
    write_json(root/('dynamic_filter_'+scene+('_flow' if flow else '')+('_derivative' if velocity_observation else '')+('_bounded' if bounded else '')+('_local_cov' if covariance_bound else '')+('_aligned' if aligned_gravity else '')+('_topology' if topology else '')+'_'+str(int(stop))+'s.json'),dict(scene=scene,fit_file='dynamic_fit_'+scene+suffix+'.json',rows=rows,failure=failure,flow_linearization=flow,velocity_observation=velocity_observation,bounded_update=bounded,covariance_bound=covariance_bound,topology_hypotheses=topology,
        wall_seconds=time.perf_counter()-begin,observer_seconds_median=float(np.median(latencies)),observer_seconds_p95=float(np.percentile(latencies,95)),limitation='Encoder/command-only EKF in private model. Initial calibration uses short active-response fit and nominal dynamics. Replay diagnostic, not a physical crossing; true geometry and root pose score only.'))
    print(scene,'failure',failure,'max axis/offset',max((r['target_error_deg'] for r in rows),default=0),max((r['target_offset_mm'] for r in rows),default=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=['horizontal','vertical'],required=True)
    p.add_argument('--stop',type=float,default=20.);p.add_argument('--flow',action='store_true');p.add_argument('--velocity-observation',action='store_true');p.add_argument('--bounded-update',action='store_true')
    p.add_argument('--covariance-bound',action='store_true');p.add_argument('--aligned-gravity',action='store_true');p.add_argument('--topology',action='store_true');a=p.parse_args();run(a.scene,a.stop,a.flow,a.velocity_observation,a.bounded_update,a.covariance_bound,a.aligned_gravity,a.topology)
