"""Rejudge saved probes; no physical rollout or true-contact gate inputs."""
import json
import mujoco
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,write_json
from feedback_takeover_response_gate import judge


def run():
    root=ROOT/'experiments/segmented_pole_20261005'
    natural=root/'natural_takeover';feedback=root/'feedback_takeover'
    cases=[('manual_horizontal',HORIZONTAL,feedback/'horizontal_response_resume_intent'),
           ('manual_vertical',VERTICAL,feedback/'vertical_response_finite_section'),
           ('natural_early',natural/'horizontal_natural_v1/candidate',natural/'horizontal_natural_v1/ensemble_cache'),
           ('natural_slow',natural/'horizontal_natural_rate_matched/candidate',natural/'horizontal_natural_rate_topology')]
    rows=[]
    for name,checkpoint,probe in cases:
        robot=mujoco.MjModel.from_binary_path(str(checkpoint/'model.mjb'))
        meta=json.loads((checkpoint/'checkpoint.json').read_text(encoding='utf8'))
        fit=json.loads((probe/'response_calibration.json').read_text(encoding='utf8'))
        records=json.loads((probe/'response_probe.json').read_text(encoding='utf8'))
        gate=judge(robot,meta['source_model'],fit,records)
        rows.append(dict(name=name,gate=gate))
        print(name,gate['criteria'],gate['distributed_support_fraction_last_point6_seconds'],flush=True)
    write_json(natural/'distributed_takeover_gate_development.json',dict(rows=rows,development_only=True,
        threshold_source='Existing grip begins above 90 deg. Same 0.4 s, >25% contact occupancy as transfer support; distributed coverage in >=90% of the last 0.6 s.'))


if __name__=='__main__':run()
