"""Read third-round saved evidence; no new MuJoCo dynamics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/unified_curve_transfer_20261007/KNOWN90_FULL_ATTEMPT_20261011"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze(case):
    z = np.load(case / "trace.npz")
    result = json.loads((case / "result.json").read_text(encoding="utf-8"))
    commands = json.loads((case / "commands.json").read_text(encoding="utf-8"))
    route = json.loads((case / "route.json").read_text(encoding="utf-8"))
    t = z["t_s"]
    tip = z["actual_points_mm"][:, 0]
    entry = np.asarray(route["target_entry_m"]) * 1000
    distance = np.linalg.norm(tip-entry, axis=1)
    lengths = np.linalg.norm(np.diff(z["execution_points_mm"], axis=1), axis=2)
    actual_lengths = np.linalg.norm(np.diff(z["actual_points_mm"], axis=1), axis=2)
    force = z["diagnostic_contact_force_N"][:, 1].sum(axis=1)
    first_force = np.flatnonzero(force > .01)
    phase = np.unwrap(np.arctan2(tip[:, 2]-1000, -tip[:, 1]))
    coeff = z["bridge_coeff_mm"][:, 0, 1:]
    head_error = np.linalg.norm(tip-z["execution_points_mm"][:, 0], axis=1)
    minimum = int(distance.argmin())
    near_end = t >= t[-1]-30.0001
    indices = np.flatnonzero(near_end)
    shift = np.asarray([c["shared_shift_mm"] for c in commands])
    same_shift = np.flatnonzero(abs(shift-shift[-1]) < 1e-7)
    summary = dict(
        actual_seconds=result["actual_seconds"], outcome=result["outcome"],
        active_capture=result["active_capture"], full_success=result["full_success"],
        maximum_target_prefix=result["maximum_actual_target_prefix"],
        maximum_target_wrap_rad=result["maximum_actual_wrap_rad"],
        maximum_target_normal_force_sum_N=float(force.max()),
        first_target_group_force_over_0p01N_s=float(t[first_force[0]]) if len(first_force) else None,
        first_target_contact_module_ids=np.flatnonzero(z["diagnostic_contact_force_N"][first_force[0], 1] > .001).tolist() if len(first_force) else [],
        final_head_tip_mm=tip[-1].tolist(),
        actual_head_orbit_ccw_deg=float(np.degrees(phase[-1]-phase[0])),
        minimum_entry_distance_mm=float(distance[minimum]), minimum_entry_distance_time_s=float(t[minimum]),
        final_entry_distance_mm=float(distance[-1]),
        last30_maximum_target_prefix=int(z["actual_target_prefix"][near_end].max()),
        last30_maximum_wrap_rad=float(z["actual_target_wrap_rad"][near_end].max()),
        last30_head_orbit_range_deg=np.degrees([phase[near_end].min()-phase[0], phase[near_end].max()-phase[0]]).tolist(),
        last30_initial_final_entry_distance_mm=[float(distance[indices[0]]), float(distance[-1])],
        final_controller_stage=commands[-1]["stage"], final_shared_shift_mm=float(shift[-1]),
        final_shift_constant_from_s=float(commands[same_shift[0]]["t_s"]),
        final_head_execution_error_mm=float(head_error[-1]),
        final_global_rms_mm=result["final_global_rms_mm"],
        final_execution_rms_mm=result["final_execution_rms_mm"],
        final_front10_execution_polyline_length_mm=float(lengths[-1, :10].sum()),
        final_execution_chord_minmax_mm=[float(lengths[-1].min()), float(lengths[-1].max())],
        maximum_actual_chord_error_mm=float(abs(actual_lengths-64.5).max()),
        maximum_execution_chord_error_mm=float(abs(lengths-64.5).max()),
        maximum_local_yz_coeff_mm=np.abs(coeff).max(axis=0).tolist(),
        kp_minmax=[float(z["kp_nm_per_rad"].min()), float(z["kp_nm_per_rad"].max())],
        maximum_radius_delta_mm=float(np.abs(z["radius_delta_mm"]).max()),
        maximum_actual_joint_deg=result["maximum_actual_joint_deg"], maximum_torque_nm=result["maximum_torque_nm"],
        maximum_pipe_depth_mm=result["maximum_pipe_depth_mm"],
        maximum_pipe_depth_time_s=float(t[z["diagnostic_pipe_depth_mm"].argmax()]),
        recorded_time_pipe_depth_over3mm_s=float(np.count_nonzero(z["diagnostic_pipe_depth_mm"] > 3)*.01),
        maximum_self_depth_mm=result["maximum_self_depth_mm"], ground=result["ground"],
        checkpoint_continuation_used=result.get("checkpoint_continuation_used", False),
        physical_root_control_assignment=False,
    )
    if case.name == "T01":
        path = np.load(case / "registered_head_route.npz")
        items = []
        for requested in (60.45, 90.45):
            i = int(np.argmin(abs(t-requested)))
            ds = np.linalg.norm(path["points_m"]-tip[i]/1000, axis=1)
            k = int(ds.argmin())
            items.append(dict(t_s=float(t[i]), tip_mm=tip[i].tolist(), projected_arc_mm=float(path["arc_m"][k]*1000),
                              path_distance_mm=float(ds[k]*1000), execution_error_mm=float(head_error[i])))
        summary["false_stall_evidence"] = items
        summary["classification"] = "runner pose-error progress filter caused premature termination; not physical no-motion proof"
    if case.name == "T03":
        t1 = np.load(BASE / "T01/trace.npz")
        n = len(t1["t_s"])
        summary["T01_overlap_actual_qpos_max_difference"] = float(abs(t1["qpos"]-z["qpos"][:n]).max())
        summary["classification"] = "original controller paused shared feed; no recovered sustained upward wrap or intake; not proof crossing physically impossible"
        figure(case, z, distance, head_error, lengths, phase)
    return summary


def figure(case, z, distance, head_error, lengths, phase):
    t = z["t_s"]
    tip = z["actual_points_mm"][:, 0]
    path = np.load(case / "registered_head_route.npz")["points_m"]*1000
    fig, ax = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("T03 frozen original action: contact occurred; shared feed did not recover", fontsize=15)
    ax[0, 0].plot(-path[:, 1], path[:, 2], color=".75", label="registered full head route")
    ax[0, 0].plot(-tip[:, 1], tip[:, 2], color="black", label="actual head tip")
    ax[0, 0].scatter(-tip[[0, -1], 1], tip[[0, -1], 2], c=["green", "red"], s=30)
    ax[0, 0].set(xlabel="Screen right: -Y (mm)", ylabel="Z (mm)", title="Tail-to-head direction +X", aspect="equal")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, z["shared_shift_mm"], label="shared reference parameter")
    ax[0, 1].axhline(500, ls="--", color=".5", label="nominal end of front unwind")
    ax[0, 1].set(xlabel="Time (s)", ylabel="mm", title="Original action stayed in front-unwind stage")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(t, head_error, label="actual head / execution error")
    ax[1, 0].axhline(70, color="red", ls="--", label="original controller feed gate")
    ax[1, 0].set(xlabel="Time (s)", ylabel="mm", title="Action gate retained; runner error filter removed")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, distance, label="actual head-to-entry distance")
    ax[1, 1].set(xlabel="Time (s)", ylabel="mm", title="New approach progress credited in every phase")
    ax[1, 1].legend(fontsize=8)
    ax[2, 0].plot(t, lengths[:, :10].sum(axis=1), label="execution front-10 polyline")
    ax[2, 0].axhline(645, color="black", ls="--", label="actual ten rods: 645 mm")
    ax[2, 0].set(xlabel="Time (s)", ylabel="mm", title="Original curve length inconsistency retained")
    ax[2, 0].legend(fontsize=8)
    ax[2, 1].plot(t, z["actual_target_prefix"], label="contiguous geometric target prefix")
    ax[2, 1].set(xlabel="Time (s)", ylabel="Modules", title="At most two; no sustained upward acquisition", ylim=(-.1, 5.5))
    ax[2, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=.25)
    for a in list(ax.flat)[1:]:
        a.axvline(154.85, color="purple", ls=":", lw=.8)
    fig.text(.01, .002, "Purple line: saved-state continuation boundary. Solver warmstart was not saved; no new candidate.", fontsize=8)
    fig.savefig(case / "actual_progress_and_controller_gate.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    cases = {name: analyze(BASE/name) for name in ("T01", "T02", "T03")}
    save(BASE / "audit.json", dict(physical_steps=0, source="existing trace, commands, results only", cases=cases))
    print(json.dumps(cases, ensure_ascii=False, indent=2))
