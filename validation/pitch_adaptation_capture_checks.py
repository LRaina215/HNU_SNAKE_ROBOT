"""Captured versus uncaptured v5 calls, identical measured geometry inputs."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import time

import numpy as np

from pitch_adaptation_experiment_v5 import ROOT,ALPHA
from pitch_adaptation_training_audit import BASE,sha
from pitch_adaptation_stage0 import geometry_from_points,helix_points
from pitch_adaptation_capture import install_capture,reconstruct_limits
from snake_control.pole_compliance import ComplianceParameters,build_backbone
from snake_control.local_pitch_controller_v5 import LocalPitchController
from Adaptive_Pole_Climbing_Mujoco import HelixBackbone


def main():
    g=geometry_from_points(helix_points(85.5))
    g.update(run_id='capture-test',config_id='20',sequence=1,age_ms=0.,sample_unix_ms=10000.,feedback_usable=True,sim_time=2.2)
    params=ComplianceParameters((.01,.02),ALPHA,False);lengths=np.full(21,64.5)
    initial=build_backbone(g,params).joint_angles(0.)
    original=LocalPitchController.compute;cases=0;max_error=0.;attempts=0
    LocalPitchController.initial_backbone=HelixBackbone.from_radius_pitch(81.5,220.)
    LocalPitchController.transfer_s=2.;LocalPitchController.control_clock=None
    for method,k in [('A',0.),('B',0.),('C',0.),('C',.5),('C',1.),('D',0.),('D',.5),('D',1.)]:
        LocalPitchController.method=method;LocalPitchController.exponent=k
        LocalPitchController.fixed_alpha=np.radians(18.) if method=='B' else ALPHA
        a=LocalPitchController(params,lengths);b=LocalPitchController(params,lengths)
        q=initial.copy();r=initial.copy();calls=[]
        for i in range(16):
            geometry=copy.deepcopy(g);geometry.update(sequence=i+1,sim_time=2.2+i*.2)
            phase=-.15*(i+1);previous=-.15*i
            LocalPitchController.compute=original;LocalPitchController.records=[]
            expected=a.compute(geometry,phase,q,now_ms=10000.,reference_phase=previous,control_time_s=geometry['sim_time'])
            install_capture(LocalPitchController,calls)
            try:actual=b.compute(geometry,phase,r,now_ms=10000.,reference_phase=previous,control_time_s=geometry['sim_time'])
            finally:LocalPitchController.compute=original
            assert expected is not None and actual is not None,(method,k,a.status,b.status)
            error=float(abs(expected-actual).max());max_error=max(max_error,error);assert error==0.
            replay=reconstruct_limits(calls[-1]);assert replay['accepted']
            assert np.array_equal(a.alpha_state,b.alpha_state) and a.key==b.key
            q,r=expected,actual;attempts+=1
        geometry=copy.deepcopy(g);geometry.update(sequence=17,sample_unix_ms=0.)
        saved=b.alpha_state.copy();install_capture(LocalPitchController,calls)
        try:rejected=b.compute(geometry,-2.55,r,now_ms=10000.,control_time_s=5.4)
        finally:LocalPitchController.compute=original
        assert rejected is None and not calls[-1]['accepted'] and np.array_equal(saved,b.alpha_state)
        json.dumps(calls,allow_nan=False);cases+=1
    result=dict(cases=cases,accepted_call_pairs=attempts,maximum_target_difference_rad=max_error,
                limit_replay_passed=True,rejected_calls_captured_without_state_change=True,synthetic_inputs_only=True,
                sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'validation/pitch_adaptation_capture.py',ROOT/'snake_control/local_pitch_controller_v5.py')})
    out=BASE/'capture_pilots';out.mkdir(parents=True,exist_ok=True)
    (out/'capture_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
