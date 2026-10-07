"""Export saved simulation increments and repository profile facts only.

No controller imports, solver, MuJoCo, socket, serial or hardware operations.
The outputs are review material, never a hardware execution package.
"""
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = ROOT / "experiments/hook_pull_review_20261007/entry_prefix_path.npz"
PROFILE = ROOT / "visualizer/profiles/20关节实机档案.json"
MODEL = ROOT / "mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml"


def main():
    with np.load(PREFIX, allow_pickle=False) as saved:
        times = saved["dispatch_time_s"].copy()
        targets = saved["dispatch_ctrl_rad"].copy()
        held = saved["held_ctrl_rad"].copy()
    assert times.shape == (41,) and targets.shape == (41, 20)
    assert np.isfinite(times).all() and np.isfinite(targets).all()
    assert np.array_equal(targets[0], held)
    assert times[0] == 0 and times[-1] == 2 and (np.diff(times) > 0).all()
    delta_deg = np.degrees(targets - targets[0])
    assert np.array_equal(delta_deg[0], np.zeros(20))
    with (HERE / "simulation_delta_deg.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + ["delta_sim_J%d_deg" % j for j in range(1, 21)])
        for time_s, row in zip(times, delta_deg):
            writer.writerow([format(float(time_s), ".17g")] +
                            [format(float(value), ".17g") for value in row])

    profile = json.loads(PROFILE.read_text(encoding="utf-8-sig"))
    xml = ET.parse(MODEL).getroot()
    joints = {j.attrib["name"]: j for j in xml.iter("joint") if "name" in j.attrib}
    actuators = list(xml.find("actuator"))
    assert len(profile["Joints"]) == len(actuators) == 20
    with (HERE / "joint_interface.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation_J", "simulation_joint", "simulation_local_axis",
                         "repository_protocol_joint_index", "repository_servo_id",
                         "repository_axis_type", "simulation_rad_to_protocol_deg_sign",
                         "repository_zero_raw", "repository_direction", "minimum_protocol_deg",
                         "maximum_protocol_deg", "link_length_mm", "field_mapping_confirmed"])
        for i, joint in enumerate(profile["Joints"]):
            name = "joint%d" % (i + 1)
            assert actuators[i].attrib["joint"] == name
            writer.writerow([i + 1, name, joints[name].attrib["axis"], i + 1,
                             joint["ServoId"], ("Yaw" if joint["AxisType"] == 0 else "Pitch"),
                             1 if i % 2 == 0 else -1, joint["Zero"], joint["Direction"],
                             joint["MinimumAngle"], joint["MaximumAngle"], joint["LinkLength"],
                             "NOT_CONFIRMED"])

    # Export integrity only; this does not repeat path/dynamics qualification.
    rows = np.loadtxt(HERE / "simulation_delta_deg.csv", delimiter=",", skiprows=1, encoding="utf-8-sig")
    assert np.array_equal(rows[:, 0], times)
    assert np.array_equal(rows[:, 1:], delta_deg)
    metadata = dict(
        date="2026-10-07", purpose="OFFLINE_REVIEW_ONLY_NOT_HARDWARE_COMMANDS",
        source=str(PREFIX.relative_to(ROOT)).replace("\\", "/"),
        source_arrays=["dispatch_time_s", "dispatch_ctrl_rad", "held_ctrl_rad"],
        definition="delta_sim_J_deg(t) = degrees(dispatch_ctrl_rad(t) - dispatch_ctrl_rad(0))",
        rows=41, simulation_joint_count=20, first_time_s=float(times[0]), last_time_s=float(times[-1]),
        original_times_preserved=True, csv_roundtrip_exact=True,
        zero_row_exact=True, units="degree increments in simulation joint coordinates",
        maximum_absolute_increment_deg=float(abs(delta_deg).max()),
        final_increment_deg=delta_deg[-1].tolist(),
        field_mapping="NOT_CONFIRMED; sign column is repository bridge convention only",
        profile_source=str(PROFILE.relative_to(ROOT)).replace("\\", "/"),
        profile_port="Repository COM9/115200 is historical configuration, not authorization or discovery",
        hardware_execution_ready=False, hardware_connections=0, new_dynamics_trials=0,
        dynamics_trials_used_this_review=2, dynamics_trial_limit_this_review=4,
        remaining_dynamics_enabled=False,
        not_exported=["absolute simulation ctrl", "root pose", "actual simulation qpos", "servo raw targets"],
    )
    (HERE / "offline_material.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("rows", "zero_row_exact", "csv_roundtrip_exact",
                                             "maximum_absolute_increment_deg", "final_increment_deg",
                                             "hardware_execution_ready", "new_dynamics_trials")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
