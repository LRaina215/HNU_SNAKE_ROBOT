"""Preserve the unchanged kinematic pre-wrap prefix for independent evaluation."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';source=B/'KINEMATIC_FULL_APPROACH_WRAP';z=np.load(source/'KINEMATIC_CONTACT_REPLAY.npz');report=json.loads((source/'REPORT.json').read_text())
last=int(np.flatnonzero(z['beta']>0.)[0]);out=B/'PRECONTACT_EVALUATION';out.mkdir(exist_ok=False)
np.savez_compressed(out/'PRECONTACT_PREFIX.npz',t=z['t'][:last],alpha=z['alpha'][:last],qpos=z['qpos'][:last],G=z['G'][:last])
rows=report['rows'][:last];summary=dict(simulation_steps=0,source=str(source),first_unlabeled_front_contact_t=report['first_unlabeled_front_contact_t'],last_preswitch_time_s=float(z['t'][last-1]),
    source_M8to20_fixed_maximum_mm=max(r['source_node_motion_mm'] for r in rows),maximum_original_depths_mm=np.max([r['original_depths_mm'] for r in rows],axis=0).tolist(),
    maximum_C_node_step_mm=max(r['C_step_mm'] for r in rows),maximum_joint_step_deg=max(r['joint_step_deg'] for r in rows),
    maximum_head_goal_error_mm=1000*max(r['head_error_m'] for r in rows),
    scope='Ideal-C kinematic source-held pre-wrap prefix only, including first geometric touch. Release clock starts at0; no physical source-settle interval or integration.')
(out/'PREFIX_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
