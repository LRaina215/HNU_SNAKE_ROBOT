"""Collect a small joint probe through the same EXE relay as rolling commands."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from segmented_pole_experiment import write_json
from snake_control.three_segment_compliance import ThreeSegmentOptions


def collect_response(sim,initial_tracker,folder):
    g=sim.fresh();q0=np.radians(g['angles_deg'])*sim.bridge.signs
    estimate=initial_tracker.estimate;points=initial_tracker.points(q0)[1:-1]
    source_distance=np.linalg.norm(np.cross(points-estimate['corner_mm'],estimate['source']),axis=1)
    target_distance=np.linalg.norm(np.cross(points-estimate['corner_mm'],estimate['target']),axis=1)
    front=min(range(2,len(q0)-ThreeSegmentOptions().minimum_rear_joints-1),
        key=lambda k:target_distance[:k].sum()+source_distance[k:].sum())
    patterns=[]
    for parity in (0,1):
        index=front+(parity-front)%2;delta=np.zeros(len(q0));delta[index]=np.radians(.6);patterns.append(delta)
    contraction=q0.copy();contraction[front:]=0.;contraction*=np.radians(.6)/max(abs(contraction).max(),1e-12)
    patterns.append(contraction);patterns=np.asarray(patterns)
    reference=q0+np.clip(sim.d.ctrl-q0,sim.m.actuator_forcerange[:,0]/sim.m.actuator_gainprm[:,0],sim.m.actuator_forcerange[:,1]/sim.m.actuator_gainprm[:,0])
    bounds=sim.m.jnt_range[sim.jids];records=[];sim.stage='active_response_probe';phase=sim.phase
    write_json(folder/'response_probe_plan.json',dict(front=front,patterns=patterns,reference=reference,phase=phase,
        amplitude_deg=.6,sample_seconds=.02,duration=1.2,commands=60,
        inputs='Relay joint encoders, previously applied commands and encoder-inferred geometry only. No physical pose/contact/axis in probe selection.'))
    for k in range(61):
        t=.02*k;g=sim.fresh();q=np.radians(g['angles_deg'])*sim.bridge.signs
        pulse=np.sin(2*np.pi*np.arange(1,4)*(t-.2)/.8) if .2<=t<=1. else np.zeros(3)
        command=np.clip(reference+pulse@patterns,bounds[:,0]+.01,bounds[:,1]-.01)
        records.append(dict(time=t,q=q,command=command,execution_time=float(sim.d.time)))
        if k<60:
            sim.command(command,g);sim.advance(.02)
    write_json(folder/'response_probe.json',records)
    return records
