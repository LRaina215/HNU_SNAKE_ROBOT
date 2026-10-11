"""Saved-data figures and observed feedback audit; no additional dynamics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from snake_control.approved_route_transfer import geometry
from validation.inner_ccw_compliance_experiment import write

BASE = ROOT/'experiments/unified_curve_transfer_20261007/KNOWN90_APPROVED_ROUTE_20261011'
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False})


def fits():
    names = [('initial', '全身原管初态'), ('leaving', '内侧出管'), ('upward', '内侧向上包绕'), ('final', '全身目标螺旋')]
    fig = plt.figure(figsize=(17, 5.8))
    surfaces = geometry.pipe_surfaces()
    for column, (name, title) in enumerate(names):
        with np.load(BASE/f'fit_{name}.npz') as z:
            reference, actual = z['reference_m']*1000, z['actual_m']*1000
        ax = fig.add_subplot(1, 4, column+1, projection='3d')
        geometry.setup3d(ax, surfaces)
        ax.plot(*reference.T, color='#de8721', lw=1.8, label='执行参考，真实杆长')
        ax.plot(*actual.T, '-o', color='#27343a', lw=2.1, ms=2, label='机构拟合，尚未运动')
        ax.scatter(*actual[0], color='red', s=25)
        ax.scatter(*actual[-1], color='blue', marker='x', s=25)
        rms = np.sqrt(np.mean(np.sum((actual-reference)**2, axis=1)))
        ax.set_title(f'{title}\n节点拟合 RMS={rms:.1f} mm', fontsize=11)
        if column == 0:ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('必要机构拟合｜20个交替关节、21根64.5 mm杆；此图不表示实际运动成功', fontsize=16)
    fig.subplots_adjust(left=.02, right=.98, top=.85, bottom=.08, wspace=.03)
    fig.savefig(BASE/'mechanism_fit_overview.png', dpi=155)
    plt.close(fig)


def candidate(case):
    if not (case/'trace.npz').exists():return
    with np.load(case/'trace.npz') as z:
        rows = {key:z[key] for key in z.files}
    commands = json.loads((case/'commands.json').read_text(encoding='utf-8'))
    t = rows['t_s']
    actual, local, global_p = rows['actual_points_mm'], rows['execution_points_mm'], rows['global_points_mm']
    entry_distance = np.linalg.norm(actual[:, 0]-geometry.E*1000, axis=1)
    nearest = int(entry_distance.argmin())
    contact = rows['diagnostic_contact_force_N'][:, 1]
    first = np.flatnonzero(contact.sum(axis=1) > .01)
    coefficients = np.asarray([c['bridge_coeff_mm'] for c in commands])
    tracking = np.asarray([c['tracking_deg'] for c in commands])
    projection = np.asarray([c['projection_rms_mm'] for c in commands])
    source_radii = np.linalg.norm(rows['actual_centers_mm'][:, :, [1, 2]]-[0, 1000], axis=2)
    target_radii = np.linalg.norm(rows['actual_centers_mm'][:, :, [0, 2]]-[1100, 1000], axis=2)
    reference_lengths = np.linalg.norm(np.diff(local, axis=1), axis=2)
    physical_lengths = np.linalg.norm(np.diff(actual, axis=1), axis=2)
    post = np.degrees(rows['actual_source_ccw_rad'][-1, 12:])
    theta = np.unwrap(np.arctan2(rows['actual_centers_mm'][:, :, 2]-1000, -rows['actual_centers_mm'][:, :, 1]), axis=0)
    source_geometry = (source_radii > 74) & (source_radii < 118) & (rows['actual_centers_mm'][:, :, 0] < 1035)
    geometry_orbit = np.sum(np.where(source_geometry[1:] & source_geometry[:-1], np.diff(theta, axis=0), 0), axis=0)
    global_centers = (global_p[:, :-1]+global_p[:, 1:])/2
    global_theta = np.unwrap(np.arctan2(global_centers[:, :, 2]-1000, -global_centers[:, :, 1]), axis=0)
    anchor_indices = np.asarray([command['anchor_module']+1 for command in commands], dtype=int)
    predicted = np.asarray([command['predicted_points_mm'] for command in commands])
    command_times = np.asarray([command['t_s'] for command in commands])
    row_indices = np.searchsorted(t, command_times-1e-7)
    row_indices = np.minimum(row_indices, len(t)-1)
    command_ids = np.arange(len(commands))
    anchor_predicted = predicted[command_ids, anchor_indices]
    anchor_actual = actual[row_indices, anchor_indices]
    commanded_execution = np.asarray([command['execution_points_mm'] for command in commands])
    anchor_reference = commanded_execution[command_ids, anchor_indices]
    private_move = np.linalg.norm(anchor_predicted-anchor_actual, axis=1)
    anchor_mismatch = np.linalg.norm(anchor_reference-anchor_actual, axis=1)
    last_target = np.asarray(commands[-1]['ctrl_rad'])
    qtargets = np.asarray([command['ctrl_rad'] for command in commands])
    unique_anchors, frequencies = np.unique(anchor_indices-1, return_counts=True)
    front_leave = 0
    while front_leave < 21 and not source_geometry[-1, front_leave]:front_leave += 1
    contact_leave = 0
    while contact_leave < 21 and not rows['diagnostic_source_contact_any'][-1, contact_leave]:contact_leave += 1
    final_stations = np.asarray(commands[-1]['material_stations_mm'])
    station_mid = (final_stations[:-1]+final_stations[1:])/2
    feed_break = dict(physical_steps_added=0, head_shift_end_mm=float(rows['shared_shift_mm'][-1]),
                      reference_M14_axial_advance_mm=float(global_centers[-1, 14, 0]-global_centers[0, 14, 0]),
                      actual_M14_axial_advance_mm=float(rows['actual_centers_mm'][-1, 14, 0]-rows['actual_centers_mm'][0, 14, 0]),
                      reference_M14_orbit_deg=float(np.degrees(global_theta[-1, 14]-global_theta[0, 14])),
                      actual_M14_orbit_deg=float(np.degrees(theta[-1, 14]-theta[0, 14])),
                      actual_target_prefix_end=int(rows['actual_target_prefix'][-1]),
                      execution_transition_rods_end=int(np.count_nonzero((station_mid >= 0) & (station_mid < geometry.Route().bridge_length*1000))),
                      actual_front_outside_source_geometry_prefix_rods=front_leave,
                      actual_front_no_source_contact_prefix_rods=contact_leave,
                      departure_metric_limitation='Geometric band74..118mm/X<1035 and source contact absence are diagnostics, not proof of an unwrapped lower transition or target capture.',
                      anchor_modules={str(a):int(n) for a,n in zip(unique_anchors, frequencies)},
                      final_anchor_module=int(anchor_indices[-1]-1),
                      maximum_private_anchor_vs_actual_increment_mm=float(private_move.max()),
                      final_private_anchor_vs_actual_increment_mm=float(private_move[-1]),
                      final_anchor_actual_to_execution_reference_mm=float(anchor_mismatch[-1]),
                      maximum_anchor_actual_to_execution_reference_mm=float(anchor_mismatch.max()),
                      rear_joints=list(range(13, 21)),
                      rear_joint_target_span_deg=np.degrees(np.ptp(qtargets[:, 12:], axis=0)),
                      rear_final_target_vs_actual_deg=np.degrees(last_target[12:]-rows['q'][-1, 12:]),
                      final_commanded_limit_joints=(np.flatnonzero(abs(last_target)>np.radians(87.99))+1).tolist(),
                      explanation='The registered bearing origin is independent of the hinge variables: its position-error Jacobian is zero. Shared shift moves reference material points, but those accumulating bearing-position errors cannot be reduced by the current private registered hinge fit.')
    write(case/'feed_breakpoint_audit.json', feed_break)
    audit = dict(physical_steps_added=0, actual_time_s=float(t[-1]),
                 closest_entry_time_s=float(t[nearest]), closest_entry_distance_mm=float(entry_distance[nearest]),
                 closest_entry_head_mm=actual[nearest, 0], final_head_mm=actual[-1, 0],
                 initial_actual_head_mm=actual[0, 0], initial_global_head_mm=global_p[0, 0],
                 initial_head_reference_error_mm=float(np.linalg.norm(actual[0, 0]-global_p[0, 0])),
                 first_true_target_contact_s=float(t[first[0]]) if len(first) else None,
                 first_target_contact_modules=np.flatnonzero(contact[first[0]] > .001).tolist() if len(first) else [],
                 maximum_target_normal_sum_N=float(contact.sum(axis=1).max()),
                 final_source_contact_modules=np.flatnonzero(rows['diagnostic_source_contact_any'][-1]).tolist(),
                 final_geometric_target_modules=np.flatnonzero((target_radii[-1] > 73) & (target_radii[-1] < 118) & (rows['actual_centers_mm'][-1, :, 1] > 150)).tolist(),
                 final_source_radii_mm=source_radii[-1], final_target_radii_mm=target_radii[-1],
                 source_contact_sampled_ccw_posterior_median_deg=float(np.median(post)),
                 source_contact_sampled_ccw_posterior_positive_fraction=float(np.mean(post > 0)),
                 source_geometry_sampled_orbit_deg=np.degrees(geometry_orbit),
                 source_geometry_sampled_posterior_median_deg=float(np.median(np.degrees(geometry_orbit[12:]))),
                 same_M14_planned_orbit_deg=float(np.degrees(global_theta[-1, 14]-global_theta[0, 14])),
                 same_M14_actual_orbit_deg=float(np.degrees(theta[-1, 14]-theta[0, 14])),
                 maximum_joint_tracking_deg=float(tracking.max()), maximum_projection_rms_mm=float(projection.max()),
                 maximum_local_bridge_coefficient_mm=float(abs(coefficients).max()),
                 local_shape_active_commands=int(np.count_nonzero(abs(coefficients).max(axis=(1, 2)) > .1)),
                 total_commands=len(commands),
                 maximum_local_vs_global_node_rms_mm=float(np.sqrt(np.mean(np.sum((local-global_p)**2, axis=2), axis=1)).max()),
                 final_actual_global_rms_mm=float(np.sqrt(np.mean(np.sum((actual[-1]-global_p[-1])**2, axis=1)))),
                 final_actual_local_rms_mm=float(np.sqrt(np.mean(np.sum((actual[-1]-local[-1])**2, axis=1)))),
                 maximum_execution_rod_length_error_mm=float(abs(reference_lengths-64.5).max()),
                 maximum_actual_rod_length_error_mm=float(abs(physical_lengths-64.5).max()),
                 control_comparison_performed=False,
                 feedback_conclusion='Activity and curve displacement can be observed; causal benefit needs the authorized successful feedback-off comparison.')
    write(case/'observed_audit.json', audit)
    indices = np.unique([0, min(len(t)-1, int(np.searchsorted(t, 30))), nearest, len(t)-1])
    with (case/'curve_snapshots.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'shared_shift_mm', 'node_id', 'curve', 'x_mm', 'y_mm', 'z_mm'])
        for i in indices:
            for points, label in [(global_p, 'global_reference'), (local, 'local_execution'), (actual, 'actual_body')]:
                for node_id, point in enumerate(points[i]):
                    writer.writerow([f'{t[i]:.3f}', f'{rows["shared_shift_mm"][i]:.6f}', node_id, label, *[f'{v:.6f}' for v in point]])
    fig = plt.figure(figsize=(4.8*len(indices), 6.0))
    surfaces = geometry.pipe_surfaces()
    for column, i in enumerate(indices):
        ax = fig.add_subplot(1, len(indices), column+1, projection='3d')
        geometry.setup3d(ax, surfaces)
        for p, color, label in [(global_p[i], '#735ca4', '全局参考'), (local[i], '#df892b', '局部执行参考'), (actual[i], '#202d37', '实际身体')]:
            ax.plot(*p.T, color=color, lw=1.8, label=label)
        ax.scatter(*actual[i, 0], color='red', s=28)
        ax.scatter(*actual[i, -1], color='blue', marker='x', s=30)
        ax.set_title(f't={t[i]:.2f}s，shift={rows["shared_shift_mm"][i]:.1f}mm\n实际目标前缀 {rows["actual_target_prefix"][i]}/21', fontsize=11)
        if column == 0:ax.legend(loc='lower left', fontsize=9)
    fig.suptitle(f'{case.name}｜同一身体节点对应的全局参考、局部执行参考与实际身体', fontsize=16)
    fig.text(.5, .035, '红点=实际头部，蓝叉=实际尾部；图中实际曲线来自连续动力学记录，参考线不代表已经到达。', ha='center', fontsize=11)
    fig.subplots_adjust(left=.02, right=.98, top=.83, bottom=.11, wspace=.02)
    fig.savefig(case/'global_local_actual_3d.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ct = np.asarray([c['t_s'] for c in commands])
    axes[0].plot(t, rows['shared_shift_mm'], label='共同参考推进量 / mm')
    axes[0].plot(t, rows['actual_target_prefix']*64.5, label='实际目标前缀杆长 / mm')
    axes[0].plot(t, entry_distance, label='实际头到固定入口距离 / mm')
    for p, label in [(global_p, '实际—全局 RMS / mm'), (local, '实际—局部 RMS / mm')]:
        axes[1].plot(t, np.sqrt(np.mean(np.sum((actual-p)**2, axis=2), axis=1)), label=label)
    axes[2].plot(ct, tracking, label='最大关节跟随误差 / °')
    axes[2].plot(ct, abs(coefficients).max(axis=(1, 2)), label='局部调形最大系数 / mm')
    axes[3].plot(t, rows['diagnostic_contact_force_N'][:, 0].sum(axis=1), label='原管法向力合计 / N')
    axes[3].plot(t, contact.sum(axis=1), label='目标管法向力合计 / N')
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=.22)
    axes[-1].set_xlabel('实际仿真时间 / s')
    fig.suptitle(f'{case.name}｜反馈作用与实际进展分开记录；接触法向力不等于竖直支撑力', fontsize=14)
    fig.tight_layout()
    fig.savefig(case/'feedback_and_progress.png', dpi=145)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    # One fixed material rod M14 (zero-based), not different rods as time grows.
    rod = 14
    center = rows['actual_centers_mm'][:, rod]
    selection = np.flatnonzero(source_geometry[:, rod])
    ax = axes[0]
    ax.add_patch(plt.Circle((0, 0), 56.5, facecolor='#b7c0c7', edgecolor='#6b7b87', alpha=.6))
    if len(selection):
        trajectory = np.column_stack([-center[selection, 1], center[selection, 2]-1000])
        ax.plot(*trajectory.T, color='#26333f', lw=1.4)
        ax.scatter(*trajectory[0], color='#7556a6', s=50, label='实际起点')
        ax.scatter(*trajectory[-1], color='#d43f42', s=50, label='实际终点（原管附近）')
        for lo, hi in zip(np.linspace(0, len(selection)-1, 5, dtype=int)[:-1], np.linspace(0, len(selection)-1, 5, dtype=int)[1:]):
            if hi > lo:ax.annotate('', xy=trajectory[hi], xytext=trajectory[lo], arrowprops=dict(arrowstyle='->', color='#26333f'))
    ax.set(xlim=(-135, 135), ylim=(-135, 135), aspect='equal', xlabel='画面右 = −Y / mm', ylabel='画面上 = Z−1000 / mm')
    ax.set_title('实际同一杆 M14：尾看头 +X 视角')
    ax.legend(fontsize=9)
    ax.grid(alpha=.22)
    for rod in (12, 14, 16, 18, 20):
        axes[1].plot(t, np.degrees(theta[:, rod]-theta[0, rod]), label=f'实际同一杆 M{rod} 原始绕行角')
    axes[1].plot(t, np.degrees(global_theta[:, 14]-global_theta[0, 14]), '--', color='#c3803a', lw=1.5, label='同一杆 M14 的规划角（尚未实现）')
    axes[1].axhline(0, color='gray', lw=.7)
    axes[1].set(xlabel='实际仿真时间 / s', ylabel='相对初始绕行角 / °', title='正角为逆时针；离开原管后的角度不用于送身判定')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.22)
    fig.suptitle(f'{case.name}｜箭头由同一材料杆的实际位置计算，未用规划箭头替代', fontsize=14)
    fig.tight_layout()
    fig.savefig(case/'actual_source_direction.png', dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, global_centers[:, 14, 0]-global_centers[0, 14, 0], label='同一后部杆M14参考轴向推进 / mm')
    axes[0].plot(t, rows['actual_centers_mm'][:, 14, 0]-rows['actual_centers_mm'][0, 14, 0], label='同一后部杆M14实际轴向推进 / mm')
    axes[1].plot(t, np.degrees(global_theta[:, 14]-global_theta[0, 14]), label='M14参考逆时针角 / °')
    axes[1].plot(t, np.degrees(theta[:, 14]-theta[0, 14]), label='M14实际绕行角 / °')
    axes[2].plot(command_times, anchor_mismatch, label='承载点实际—执行参考间距 / mm')
    axes[2].plot(command_times, private_move, label='私有拟合允许的承载点相对实际位移 / mm')
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('实际仿真时间 / s')
    fig.suptitle(f'{case.name}｜共同shift没有自动变成真实后部供身：材料点与承载注册的断点', fontsize=14)
    fig.tight_layout()
    fig.savefig(case/'rear_feed_breakpoint.png', dpi=145)
    plt.close(fig)


if __name__ == '__main__':
    fits()
    for case in BASE.iterdir():
        if case.is_dir() and (case/'trace.npz').exists():candidate(case)
