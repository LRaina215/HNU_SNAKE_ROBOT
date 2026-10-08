"""Check saved receiving deployment, without advancing physical dynamics."""
from pathlib import Path
import argparse,copy,json,sys
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from snake_control.moving_compliance_transfer import MovingComplianceTransfer
from snake_control.hook_pull_review_support import restore_saved_state
from diagnose_explicit_winding_stop import minimum


def stats(controller,shift):
    nodes,value=minimum(controller.curve,controller.pipe_geometry,shift)
    rod=value['limiting_rod'];mid=float(np.mean(controller.curve.reference_node_parameters[rod:rod+2]))
    spec=controller.curve.planned_route_spec
    value.update(limiting_rod_material_midstation_mm=mid*1000,
        inside_original_bridge_avoidance_scope=bool(spec['source_bound']-.0645<=mid<=spec['target_bound']+.0645))
    return value


def diagnose(case):
    model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    data=mujoco.MjData(model)
    restore_saved_state(model,data,ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz')
    controller=MovingComplianceTransfer(model,data)
    initial=copy.deepcopy(controller.curve.__dict__)
    controller.load_state(case/'controller_state.json')
    s=controller.shift;front=controller.front_local_material;rear=controller.rear_local_material
    report=dict(name=case.name,no_physics_steps=True,common_window_mm=s*1000,
        local_front_rear_mm=[front*1000,rear*1000],saved_final_reference=stats(controller,s),
        initial_to_corridor_maximum_captured_residual_mm=float(np.linalg.norm(np.asarray(initial['planned_route_spec']['entry_residual']),axis=1).max()*1000))
    controller.curve.__dict__.clear();controller.curve.__dict__.update(initial)
    controller.curve.increment(s);controller.set_local_curve(front,rear)
    report['same_progress_without_pressure_or_clearance_fields']=stats(controller,s)
    try:controller.route_outside_bridge(s);error=None
    except ValueError as exc:error=str(exc)
    report['bounded_bridge_avoidance']=stats(controller,s)
    report['bounded_bridge_avoidance_error']=error
    report['bounded_bridge_avoidance_field_mm']=float(np.linalg.norm(controller.curve.bridge_clearance_offsets,axis=1).max()*1000)
    report['meaning']='captured initial Cartesian residual deploys the new corridor through source rods; original bridge-only avoidance can move an intrusion to an uncovered source rod; no actual-motion or shell-safety conclusion'
    (case/'receiving_path_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);args=p.parse_args();diagnose(args.case)
