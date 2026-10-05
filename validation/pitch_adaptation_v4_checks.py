"""Regression checks for physical timing/cache and production-rule preservation."""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import hashlib
import json
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'validation')]
from snake_control import local_pitch_controller_v4 as controller
from snake_control.pole_compliance import ComplianceParameters,build_backbone
from pitch_adaptation_stage0 import ALPHA,LINK,JOINTS,geometry_from_points,helix_points


def main():
    out=ROOT/'experiments/pitch_adaptation_20261003/repair_v4'
    out.mkdir(parents=True,exist_ok=True)
    g=geometry_from_points(helix_points(85.5))
    g.update(run_id='regression',config_id='20',sequence=1,age_ms=0.,sample_unix_ms=time.time()*1000,
             feedback_usable=True,sim_time=2.2)
    params=ComplianceParameters((.01,.02),ALPHA,False)
    reference=build_backbone(g,params).joint_angles(0.)
    controller.LocalPitchController.method='A';controller.LocalPitchController.records=[]
    controller.LocalPitchController.transfer_s=0.
    a=controller.LocalPitchController(params,np.full(JOINTS+1,LINK))
    q=a.compute(g,-.15,reference,reference_phase=0.,control_time_s=2.2)
    assert q is not None,a.status
    expected=build_backbone(g,params).joint_angles(-.15)
    baseline_error=float(abs(q-expected).max());assert baseline_error<=1e-12
    controller.LocalPitchController.method='C'
    c=controller.LocalPitchController(params,np.full(JOINTS+1,LINK))
    q=c.compute(g,-.15,reference,reference_phase=0.,control_time_s=2.2)
    assert q is not None,c.status
    radius=c.global_radius
    estimate=controller.estimate_module_radius
    def invalid(*args,**kwargs):
        result=estimate(*args,**kwargs)
        for row in result['nodes']:row['valid']=False
        return result
    controller.estimate_module_radius=invalid
    try:
        g['sequence']+=1
        q=c.compute(g,-.30,q,reference_phase=-.15,control_time_s=2.4)
        assert q is not None,c.status
        assert c.global_radius==radius and np.isfinite(c.alpha_state).all()
        held=c.records[-1];assert all(held['radius_usable'])
        previous=c.alpha_state.copy()
        q=c.compute(g,-.30,q,reference_phase=-.30,control_time_s=2.4)
        assert q is not None,c.status
        assert np.array_equal(c.alpha_state,previous)
        g['sequence']+=1
        q=c.compute(g,-.45,q,reference_phase=-.30,control_time_s=2.8)
        assert q is not None,c.status
        assert not any(c.records[-1]['radius_usable'])
        assert np.max(abs(c.records[-1]['requested_alpha_rad']-np.full(JOINTS+2,ALPHA)))<1e-12
        state=c.alpha_state.copy();old_key=c.key
        g['sample_unix_ms']=time.time()*1000-1000
        assert c.compute(g,-.6,q,control_time_s=3.) is None
        assert np.array_equal(state,c.alpha_state) and c.key==old_key
        g['sample_unix_ms']=time.time()*1000;g['sequence']-=1
        assert c.compute(g,-.6,q,control_time_s=3.) is None
        assert np.array_equal(state,c.alpha_state) and c.key==old_key
        g['run_id']='new-run';g['sequence']=1
        q=c.compute(g,-.15,reference,reference_phase=0.,control_time_s=.2)
        assert q is not None,c.status
        assert c.global_radius is None and not any(c.records[-1]['radius_usable'])
    finally:controller.estimate_module_radius=estimate
    controller.LocalPitchController.method='D'
    for k in (0.,.5,1.):
        controller.LocalPitchController.exponent=k
        d=controller.LocalPitchController(params,np.full(JOINTS+1,LINK))
        g['sample_unix_ms']=time.time()*1000
        target=d.compute(g,-.15,reference,reference_phase=0.,control_time_s=.2)
        assert target is not None,d.status
        record=d.records[-1]
        active=np.asarray(record['pitch_constraint_active']);p=np.asarray(record['implied_pitch_mm'],float)
        assert np.all((p[active]>=100.-1e-8)&(p[active]<=300.+1e-8))
        assert np.max(abs(d.alpha_state-ALPHA))<=np.radians(1.)+1e-12
    controller.LocalPitchController.transfer_s=2.
    controller.LocalPitchController.method='A'
    handoff=controller.LocalPitchController(params,np.full(JOINTS+1,LINK))
    g['sample_unix_ms']=time.time()*1000
    target=handoff.compute(g,-.15,reference,reference_phase=0.,control_time_s=2.2)
    assert target is not None,handoff.status
    assert handoff.records[-1]['takeover_blend']==.1
    assert handoff.started and abs(target-reference).max()<=np.radians(30.)
    result=dict(passed=True,common_takeover_passed=True,original_fixed_alpha_target_max_error_rad=baseline_error,
                empty_current_frame_C_cache_passed=True,physical_timeout_passed=True,
                repeated_physical_time_no_filter_advance=True,stale_out_of_order_state_preserved=True,
                new_run_cache_reset=True,post_filter_pitch_bounds_all_k=True,
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),ROOT/'snake_control/local_pitch_controller_v4.py',ROOT/'snake_control/local_pitch_feedback.py')})
    (out/'controller_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
