"""Audit existing second-round evidence. This entry never steps MuJoCo."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/unified_curve_transfer_20261007/KNOWN90_GLOBAL_LOCAL_20261011"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(name):
    case = BASE / name
    z = np.load(case / "trace.npz")
    commands = json.loads((case / "commands.json").read_text(encoding="utf-8"))
    result = json.loads((case / "result.json").read_text(encoding="utf-8"))
    centers = z["actual_centers_mm"]
    radii = np.linalg.norm(centers[:, :, [1, 2]] - [0, 1000], axis=2)
    geometry = (radii > 74) & (radii < 120) & (centers[:, :, 0] < 1035)
    forces = z["diagnostic_contact_force_N"].sum(axis=2)
    head_error = np.linalg.norm(z["actual_points_mm"][:, 0] - z["execution_points_mm"][:, 0], axis=1)
    global_head_error = np.linalg.norm(z["actual_points_mm"][:, 0] - z["global_points_mm"][:, 0], axis=1)
    phase = np.unwrap(np.arctan2(centers[:, :, 2] - 1000, -centers[:, :, 1]), axis=0)
    tip = z["actual_points_mm"][:, 0]
    tip_phase = np.unwrap(np.arctan2(tip[:, 2] - 1000, -tip[:, 1]))
    posterior_wrap = np.ptp(np.unwrap(np.arctan2(centers[:, 10:, 2] - 1000, -centers[:, 10:, 1]), axis=1), axis=1)
    coeff = z["bridge_coeff_mm"][:, 0, 1:]
    last = len(z["t_s"]) - 1
    result_summary = dict(
        actual_seconds=result["actual_seconds"], outcome=result["outcome"],
        full_success=result["full_success"], active_capture=result["active_capture"],
        actual_progress_station_mm=result["actual_progress_station_mm"],
        maximum_target_prefix=result["maximum_actual_target_prefix"],
        maximum_target_wrap_rad=result["maximum_actual_wrap_rad"],
        final_global_rms_mm=result["final_global_rms_mm"],
        final_execution_rms_mm=result["final_execution_rms_mm"],
        final_head_execution_error_mm=float(head_error[last]),
        final_head_global_error_mm=float(global_head_error[last]),
        maximum_local_yz_coeff_mm=np.abs(coeff).max(axis=0).tolist(),
        local_yz_nonzero_commands=sum(float(np.abs(np.asarray(c["bridge_coeff_mm"])).max()) > .01 for c in commands),
        total_control_commands=len(commands),
        kp_min=float(z["kp_nm_per_rad"].min()), kp_max=float(z["kp_nm_per_rad"].max()),
        maximum_radius_delta_mm=float(np.abs(z["radius_delta_mm"]).max()),
        initial_source_normal_force_sum_N=float(forces[0, 0]),
        final_source_normal_force_sum_N=float(forces[last, 0]),
        maximum_target_normal_force_sum_N=float(forces[:, 1].max()),
        initial_source_contact_modules=int(z["source_support_count"][0]),
        final_source_contact_modules=int(z["source_support_count"][last]),
        final_source_contact_module_ids=np.flatnonzero(z["diagnostic_source_contact_any"][last]).tolist(),
        final_posterior7_source_geometry_count=int(geometry[last, -7:].sum()),
        final_posterior11_source_geometry_count=int(geometry[last, 10:].sum()),
        final_posterior7_radii_mm=radii[last, -7:].tolist(),
        final_posterior11_radii_minmax_mm=[float(radii[last, 10:].min()), float(radii[last, 10:].max())],
        posterior11_initial_final_spatial_wrap_rad=[float(posterior_wrap[0]), float(posterior_wrap[last])],
        actual_front10_source_geometry_initial_final=[int(geometry[0, :10].sum()), int(geometry[last, :10].sum())],
        actual_tip_initial_final_mm=[tip[0].tolist(), tip[last].tolist()],
        actual_tip_net_displacement_mm=float(np.linalg.norm(tip[last] - tip[0])),
        actual_tip_ccw_change_deg=float(np.degrees(tip_phase[last] - tip_phase[0])),
        posterior7_all_pose_orbit_change_deg=np.degrees(phase[last, -7:] - phase[0, -7:]).tolist(),
        posterior7_contact_sampled_orbit_change_deg=np.degrees(z["actual_source_ccw_rad"][last, -7:]).tolist(),
        final_source_registration_axial_mm=float(commands[-1]["source_pose_axial_mm"]),
        final_source_registration_rx_angle_deg=float(np.degrees(commands[-1]["source_pose_azimuth_rad"])),
        maximum_actual_joint_deg=result["maximum_actual_joint_deg"],
        maximum_torque_nm=result["maximum_torque_nm"],
        maximum_pipe_depth_mm=result["maximum_pipe_depth_mm"],
        maximum_self_depth_mm=result["maximum_self_depth_mm"],
        ground=result["ground"], physical_root_reset=result["physical_root_reset"],
    )
    if name == "S03_ACTUAL":
        stop = z["t_s"] >= result["actual_seconds"] - .55
        result_summary["final_0p55s_source_normal_force_sum_minmax_N"] = [float(forces[stop, 0].min()), float(forces[stop, 0].max())]
        result_summary["final_0p55s_source_contact_count_minmax"] = [int(z["source_support_count"][stop].min()), int(z["source_support_count"][stop].max())]
        route = json.loads((case / "route.json").read_text(encoding="utf-8"))
        entry = np.asarray(route["target_entry_m"]) * 1000
        result_summary["actual_tip_initial_final_entry_distance_mm"] = [float(np.linalg.norm(tip[0] - entry)), float(np.linalg.norm(tip[last] - entry))]
        diagnostic_figure(case, z, forces, geometry, coeff, head_error, posterior_wrap)
    return result_summary


def diagnostic_figure(case, z, forces, geometry, coeff, head_error, wrap):
    t = z["t_s"]
    fig, ax = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("S03 actual evidence: early unwinding, no target capture (33.55 s)", fontsize=15)
    path = np.load(case / "registered_head_route.npz")["points_m"] * 1000
    tip = z["actual_points_mm"][:, 0]
    ax[0, 0].plot(path[:, 1], path[:, 2], color="0.75", label="registered full head route")
    ax[0, 0].plot(tip[:, 1], tip[:, 2], color="black", label="actual head tip")
    ax[0, 0].scatter(tip[[0, -1], 1], tip[[0, -1], 2], c=["green", "red"], s=35, zorder=4)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax[0, 0].plot(56.5 * np.cos(theta), 1000 + 56.5 * np.sin(theta), color="0.4", ls="--", label="source pipe cross-section")
    ax[0, 0].set(xlabel="Y (mm)", ylabel="Z (mm)", title="Head follows only the early route segment", aspect="equal")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, z["source_support_count"], label="actual source contact modules")
    ax[0, 1].plot(t, geometry[:, -7:].sum(axis=1), label="last 7 in source geometry")
    ax[0, 1].axhline(4, color="red", ls="--", label="stop proxy: <4 for 0.5 s")
    ax[0, 1].set(xlabel="Time (s)", ylabel="Module count", title="Contact count and retained geometry differ", ylim=(0, 22))
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(t, forces[:, 0], label="source normal-force sum")
    ax[1, 0].plot(t, forces[:, 1], label="target normal-force sum")
    ax[1, 0].set(xlabel="Time (s)", ylabel="N", title="Scalar contact sum; not net vertical support")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, coeff[:, 0], label="local Y coefficient")
    ax[1, 1].plot(t, coeff[:, 1], label="local Z coefficient")
    ax[1, 1].set(xlabel="Time (s)", ylabel="mm", title="Feedback became active; Kp remained 127")
    ax[1, 1].legend(fontsize=8)
    ax[2, 0].plot(t, head_error, label="head execution error")
    rms = np.sqrt(np.mean(np.sum((z["execution_points_mm"] - z["actual_points_mm"]) ** 2, axis=2), axis=1))
    ax[2, 0].plot(t, rms, label="whole-body execution RMS")
    ax[2, 0].set(xlabel="Time (s)", ylabel="mm", title="Head and whole-body error are distinct")
    ax[2, 0].legend(fontsize=8)
    ax[2, 1].plot(t, wrap / (2 * np.pi), label="posterior 11 spatial turns")
    ax[2, 1].plot(t, z["actual_target_prefix"], label="target prefix / modules")
    ax[2, 1].set(xlabel="Time (s)", title="Source wrapping remained; target prefix stayed zero")
    ax[2, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=.25)
    fig.savefig(case / "unfold_support_feedback.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    cases = {name: summarize(name) for name in ("S01", "S02", "S03_ACTUAL")}
    dump(BASE / "audit.json", dict(source="existing trace.npz, commands.json and result.json only", physical_steps=0,
                                  force_interpretation="sum of scalar normal forces; not net vertical support", cases=cases))
    print(json.dumps(cases, ensure_ascii=False, indent=2))
