"""One global curve, exact 21-rod windows, planning figures only.

No controller import, joint IK, dynamics, or actual-motion claim.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from math import comb
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MODEL = ROOT / "mujoco_model/Snake_Robot_o20_full_source_inner.xml"
L, PIPE_R, BODY_R = .0645, .0565, .029
R, PITCH = .093, .180
ALPHA = PITCH/np.hypot(PITCH, 2*np.pi*R)
BETA = np.sqrt(1-ALPHA**2)
SOURCE_PHI, TARGET_PHI = -np.pi/6, -5*np.pi/6
Q = np.array([.86, R*np.cos(SOURCE_PHI), 1+R*np.sin(SOURCE_PHI)])
E = np.array([1.1+R*np.cos(TARGET_PHI), .20, 1+R*np.sin(TARGET_PHI)])
TS = np.array([ALPHA, BETA*np.sin(SOURCE_PHI), -BETA*np.cos(SOURCE_PHI)])
TE = np.array([BETA*np.sin(TARGET_PHI), ALPHA, -BETA*np.cos(TARGET_PHI)])
COLORS = ["#7556a6", "#e28b24", "#008e94"]
HEAD, TAIL = "#d43f42", "#225ca7"
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 10, "savefig.facecolor": "white"})


class Route:
    def __init__(self):
        # Fixed open lower bow. It is present from shift=0, with source and
        # receiving unit tangents at its endpoints; no front-loop homotopy.
        self.controls = np.array([Q, Q+.06*TS, [.94, .11, .80], E-.07*TE, E])
        u = np.linspace(0, 1, 6001)
        p = sum(comb(4, i)*((1-u)**(4-i)*u**i)[:, None]*self.controls[i] for i in range(5))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        self.bridge_length = float(arc[-1])
        self.bridge = CubicSpline(arc, p, axis=0)
        self.helix_step = brentq(lambda ds: np.hypot(ALPHA*ds, 2*R*np.sin(BETA*ds/(2*R)))-L, .05, .08)
        self.final_shift = self.bridge_length + 22*self.helix_step

    def source(self, s):
        s = np.atleast_1d(s)
        phi = SOURCE_PHI-BETA*s/R
        return np.column_stack([Q[0]+ALPHA*s, R*np.cos(phi), 1+R*np.sin(phi)])

    def target(self, s):
        s = np.atleast_1d(s)
        phi = TARGET_PHI-BETA*s/R
        return np.column_stack([1.1+R*np.cos(phi), E[1]+ALPHA*s, 1+R*np.sin(phi)])

    def sample(self, s):
        s = np.atleast_1d(s)
        p = self.target(s-self.bridge_length)
        source, bridge = s < 0, (s >= 0) & (s < self.bridge_length)
        p[source] = self.source(s[source])
        p[bridge] = self.bridge(s[bridge])
        return p

    def window(self, shift):
        stations, points = [float(shift)], [self.sample([shift])[0]]
        for _ in range(21):
            s, p = stations[-1], points[-1]
            grid = s-np.linspace(.0001, .20, 50)
            residual = np.linalg.norm(self.sample(grid)-p, axis=1)-L
            crossing = np.flatnonzero((residual[1:] >= 0) & (residual[:-1] < 0))
            if not len(crossing):
                raise ValueError("No preceding 64.5 mm rod chord on this curve")
            j = int(crossing[0])
            previous = brentq(lambda v: np.linalg.norm(self.sample([v])[0]-p)-L,
                               grid[j+1], grid[j], xtol=1e-12)
            stations.append(previous)
            points.append(self.sample([previous])[0])
        return np.array(points), np.array(stations)


def segment_distance(p1, q1, p2, q2):
    """Closest distance between two finite segments."""
    d1, d2, rr = q1-p1, q2-p2, p1-p2
    a, e = d1@d1, d2@d2
    b, c, f = d1@d2, d1@rr, d2@rr
    den = a*e-b*b
    s = np.clip((b*f-c*e)/den, 0, 1) if den > 1e-16 else 0.
    t = (b*s+f)/e
    if t < 0:
        t, s = 0., np.clip(-c/a, 0, 1)
    elif t > 1:
        t, s = 1., np.clip((b-c)/a, 0, 1)
    return float(np.linalg.norm(p1+s*d1-p2-t*d2))


def geometry_check(route):
    # Extended axis capsules contain the actual cylinders and mitered pieces.
    # Passing this conservative 29 mm body envelope avoids obvious pipe hits;
    # it does not replace alternating-hinge or dynamics verification.
    axes = [(np.array([-.20, 0, 1]), np.array([1.1565, 0, 1])),
            (np.array([1.1, -.0565, 1]), np.array([1.1, 2., 1]))]
    pipe_clearance, self_clearance, length_error, bend = np.inf, np.inf, 0., 0.
    for shift in np.linspace(0, route.final_shift, 81):
        p, _ = route.window(shift)
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        length_error = max(length_error, abs(lengths-L).max())
        tangents = np.diff(p, axis=0)/lengths[:, None]
        bend = max(bend, np.degrees(np.arccos(np.clip(np.sum(tangents[:-1]*tangents[1:], axis=1), -1, 1))).max())
        for i in range(21):
            for a, b in axes:
                pipe_clearance = min(pipe_clearance, segment_distance(p[i], p[i+1], a, b)-PIPE_R-BODY_R)
            for j in range(i+2, 21):
                self_clearance = min(self_clearance, segment_distance(p[i], p[i+1], p[j], p[j+1])-2*BODY_R)
    t0, t1 = route.bridge(0, 1), route.bridge(route.bridge_length, 1)
    joins = [np.degrees(np.arccos(np.clip(t0/np.linalg.norm(t0)@TS, -1, 1))),
             np.degrees(np.arccos(np.clip(t1/np.linalg.norm(t1)@TE, -1, 1)))]
    return dict(dynamics_steps=0, joint_IK_performed=False, windows_checked=81,
                physical_rods=21, rod_length_mm=64.5, physical_body_length_mm=1354.5,
                maximum_rod_length_error_mm=float(length_error*1000),
                minimum_conservative_pipe_clearance_mm=float(pipe_clearance*1000),
                minimum_nonadjacent_rod_capsule_clearance_mm=float(self_clearance*1000),
                maximum_adjacent_centerline_turn_deg=float(bend),
                join_position_error_mm=[0., 0.], join_tangent_angle_error_deg=joins,
                continuous_order="P0=head, P21=tail; strictly decreasing curve station behind head",
                clearance_model="finite segment / extended pipe-axis capsules; robot capsule radius29mm; conservative geometric envelope only")


def pipe_surfaces():
    root = ET.parse(MODEL).getroot()
    body = root.find("worldbody/body[@name='pipe_network']")
    quat = np.fromstring(body.attrib["quat"], sep=" ")
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    origin = np.fromstring(body.attrib["pos"], sep=" ")
    meshes = {m.attrib["name"]: np.fromstring(m.attrib["vertex"], sep=" ").reshape(-1, 3)
              for m in root.findall("asset/mesh") if "vertex" in m.attrib}
    surfaces = []
    for geom in body.findall("geom"):
        if geom.attrib["type"] == "mesh":
            v = meshes[geom.attrib["mesh"]]+np.fromstring(geom.attrib["pos"], sep=" ")
            v = v@rot.T+origin
            surfaces.append(v[ConvexHull(v).simplices]*1000)
        else:
            endpoints = np.fromstring(geom.attrib["fromto"], sep=" ").reshape(2, 3)@rot.T+origin
            if geom.attrib["name"] == "pipe_straight_01":
                endpoints[0, 0] = .38
            else:
                endpoints[1, 1] = .76  # display only; original straight pipe ends atY=2m
            tangent = endpoints[1]-endpoints[0]
            tangent /= np.linalg.norm(tangent)
            n = np.cross(tangent, [0, 0, 1]); n /= np.linalg.norm(n)
            b = np.cross(tangent, n)
            angle = np.linspace(0, 2*np.pi, 49)
            circle = PIPE_R*(np.cos(angle)[:, None]*n+np.sin(angle)[:, None]*b)
            v = np.array([endpoints[0]+circle, endpoints[1]+circle])*1000
            faces = [np.array([v[0, i], v[0, i+1], v[1, i+1], v[1, i]]) for i in range(48)]
            surfaces.append(faces)
    return surfaces


def setup3d(ax, surfaces, zoom=False):
    for surf in surfaces:
        ax.add_collection3d(Poly3DCollection(surf, facecolor="#94a0aa", edgecolor="none", alpha=.20))
    if zoom:
        ax.set(xlim=(790, 1210), ylim=(0, 370), zlim=(790, 1150))
        ax.set_box_aspect((420, 370, 360))
        ax.set_xticks([850, 1000, 1150]); ax.set_yticks([0, 150, 300]); ax.set_zticks([800, 950, 1100])
    else:
        ax.set(xlim=(400, 1230), ylim=(-150, 680), zlim=(780, 1210))
        ax.set_box_aspect((830, 830, 430))
        ax.set_xticks([500, 800, 1100]); ax.set_yticks([0, 300, 600]); ax.set_zticks([850, 1000, 1150])
    ax.set_xlabel("X / mm", labelpad=2); ax.set_ylabel("Y / mm", labelpad=2); ax.set_zlabel("Z / mm", labelpad=2)
    ax.view_init(elev=26, azim=-56)
    ax.tick_params(labelsize=8, pad=0)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (.8, .83, .86, .35)


def route_line(ax, route, alpha=.24):
    pieces = [np.linspace(-21*route.helix_step, 0, 600),
              np.linspace(0, route.bridge_length, 450),
              np.linspace(route.bridge_length, route.final_shift+.08, 900)]
    for s, color in zip(pieces, COLORS):
        p = route.sample(s)*1000
        ax.plot(*p.T, color=color, lw=1.05, alpha=alpha)


def body_line(ax, route, shift):
    p, s = route.window(shift)
    sm = (s[:-1]+s[1:])/2
    roles = np.where(sm < 0, 0, np.where(sm < route.bridge_length, 1, 2))
    for i, role in enumerate(roles):
        ax.plot(*(p[i:i+2]*1000).T, color=COLORS[role], lw=3.3, solid_capstyle="round")
    ax.scatter(*(p*1000).T, color="#303b48", s=5, depthshade=False)
    ax.scatter(*(p[0]*1000), color=HEAD, s=37, depthshade=False)
    ax.scatter(*(p[-1]*1000), color=TAIL, marker="x", s=37, depthshade=False)
    ax.text(*(p[0]*1000+[8, 7, 7]), "头", color=HEAD, fontsize=9, weight="bold")
    ax.text(*(p[-1]*1000+[-10, -12, 12]), "尾", color=TAIL, fontsize=9, weight="bold")
    return p, s, [int(np.count_nonzero(roles == j)) for j in range(3)]


def draw(route, surfaces):
    ds, bl = route.helix_step, route.bridge_length
    stages = [("① 全身在原管", 0.), ("② 内侧逐段出管", .55*bl),
              ("③ 从下部开始向上绕", bl+R*np.pi/3/BETA),
              ("④ 前部包绕、后部继续送身", bl+5*ds),
              ("⑤ 前后接续同一目标螺旋", bl+12*ds),
              ("⑥ 全身进入目标螺旋", route.final_shift)]
    fig = plt.figure(figsize=(18, 11.6))
    fig.suptitle("已知90°跨管｜一条完整全局参考曲线", fontsize=22, y=.982, weight="bold")
    fig.text(.5, .946, "21杆 × 64.5 mm = 1354.5 mm；共同 shift 推进，身体始终按真实杆长取点", ha="center", fontsize=12)
    fig.text(.5, .918, "全部为规划参考，尚未运行运动验证", ha="center", fontsize=11, color="#b14427")
    legend = [Line2D([0], [0], color=c, lw=3, label=l) for c, l in zip(COLORS, ["原管段", "内侧过渡段", "目标螺旋段"])]
    legend += [Line2D([0], [0], color=".65", lw=1, label="细线：完整路线"),
               Line2D([0], [0], color=".2", lw=3, label="粗杆：当前参考身体")]
    fig.legend(handles=legend, ncol=5, loc="upper center", bbox_to_anchor=(.5, .905), frameon=False, fontsize=11)
    saved = []
    for k, (label, shift) in enumerate(stages):
        ax = fig.add_subplot(2, 3, k+1, projection="3d")
        setup3d(ax, surfaces)
        route_line(ax, route)
        p, s, counts = body_line(ax, route, shift)
        ax.set_title(f"{label}\nshift = {shift*1000:.1f} mm｜原/过渡/目标：{counts[0]}/{counts[1]}/{counts[2]}杆", fontsize=12, pad=1)
        if k == 0:
            ax.text(1020, 390, 1100, "目标管", fontsize=10, color="#586674")
            ax.text(470, -90, 1020, "原管", fontsize=10, color="#586674")
        saved.append(dict(stage=k+1, label=label, shift_m=float(shift), points_m=p.tolist(), stations_m=s.tolist(), roles=counts))
    fig.subplots_adjust(left=.015, right=.985, top=.835, bottom=.07, wspace=.02, hspace=.10)
    fig.text(.5, .024, "同一坐标与视角；头=红点，尾=蓝叉；目标直管只显示局部。灰色管道尺寸来自现有模型。", ha="center", fontsize=10)
    fig.savefig(OUT/"global_route_stages.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7.2))
    fig.suptitle("内侧上绕近景｜过渡弧从原管出发，直接接入目标螺旋", fontsize=18, weight="bold", y=.97)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    setup3d(ax, surfaces, zoom=True)
    route_line(ax, route, alpha=.65)
    body_line(ax, route, stages[2][1])
    ax.text(*(Q*1000+[-55, 10, 0]), "出管Q", color=COLORS[1], fontsize=11)
    ax.text(*(E*1000+[0, 28, -30]), "下部入口E", color=COLORS[2], fontsize=11)
    up = route.target(np.array([.025, .080]))*1000
    ax.quiver(*up[0], *(up[1]-up[0]), color=COLORS[2], arrow_length_ratio=.26, linewidth=2.2)
    ax.set_title("夹角内侧：X < 1100 mm，Y > 0", fontsize=12)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.add_patch(Circle((1100, 1000), PIPE_R*1000, facecolor="#bac3ca", edgecolor="#697b89", alpha=.55))
    ax2.axvspan(980, 1100, color="#eaf5f4", zorder=-2)
    w = np.linspace(0, R*(2*np.pi/3)/BETA, 240)
    p = route.target(w)*1000
    ax2.plot(p[:, 0], p[:, 2], color=COLORS[2], lw=3)
    for lo, hi in [(18, 46), (86, 118), (168, 195)]:
        ax2.annotate("", xy=p[hi, [0, 2]], xytext=p[lo, [0, 2]], arrowprops=dict(arrowstyle="->", color=COLORS[2], lw=2.6))
    ax2.scatter(E[0]*1000, E[2]*1000, c=COLORS[1], s=50)
    ax2.annotate("从内侧下部进入", E[[0, 2]]*1000, xytext=(1000, 900), arrowprops=dict(arrowstyle="->", color=COLORS[1]), fontsize=12)
    ax2.text(1095, 1000, "目标管", ha="center", va="center", fontsize=12)
    ax2.text(1005, 1108, "沿目标螺旋向上", color=COLORS[2], fontsize=12)
    ax2.text(996, 1020, "内侧", color="#45716d", fontsize=11)
    ax2.axhline(1000, ls=":", color=".7"); ax2.axvline(1100, ls=":", color=".7")
    ax2.set(xlim=(980, 1170), ylim=(880, 1130), xlabel="X / mm", ylabel="Z / mm", aspect="equal")
    ax2.set_title("目标管横截面：低处 → 内侧 → 上部", fontsize=12)
    ax2.grid(alpha=.20)
    fig.text(.5, .035, "箭头表示计划方向；当前只检查参考几何，未证明实际搭管或承接。", ha="center", fontsize=11, color="#b14427")
    fig.subplots_adjust(left=.04, right=.96, top=.85, bottom=.12, wspace=.12)
    fig.savefig(OUT/"inner_upward_closeup.png", dpi=180)
    plt.close(fig)

    shifts = np.array([0., .10, .20, .30])
    point = np.array([route.window(v)[0][14] for v in shifts])
    theta = np.unwrap(np.arctan2(point[:, 2]-1, -point[:, 1]))
    if not np.all(np.diff(theta) > 0):
        raise ValueError("The same source node does not move CCW under shared shift")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    fig.suptitle("尾看头的逆时针方向｜追踪同一个身体节点 P14", fontsize=18, weight="bold")
    ax = axes[0]
    ax.add_patch(Circle((0, 0), PIPE_R*1000, facecolor="#bfc7ce", edgecolor="#697b89", alpha=.65))
    angle = np.linspace(0, 2*np.pi, 300)
    ax.plot(R*1000*np.cos(angle), R*1000*np.sin(angle), "--", color=".65", lw=1)
    colors = ["#7556a6", "#e28b24", "#008e94", "#d43f42"]
    for i in range(4):
        u, v = -point[i, 1]*1000, (point[i, 2]-1)*1000
        ax.scatter(u, v, color=colors[i], s=70, zorder=5)
        ax.annotate(f"shift {shifts[i]*1000:.0f}", (u, v), xytext=(u*1.20, v*1.20), color=colors[i], ha="center", fontsize=11)
        if i < 3:
            a = np.linspace(theta[i]+.025, theta[i+1]-.025, 45)
            arc = R*1000*np.column_stack([np.cos(a), np.sin(a)])
            ax.plot(*arc.T, color=".35", lw=1.5)
            ax.annotate("", xy=arc[-1], xytext=arc[-7], arrowprops=dict(arrowstyle="->", color=".35", lw=2))
    ax.text(0, 0, "原管\n视线沿+X", ha="center", va="center", fontsize=12)
    ax.set(xlim=(-135, 135), ylim=(-135, 135), aspect="equal", xlabel="画面右 = −Y / mm", ylabel="画面上 = Z−1000 / mm")
    ax.set_title("位置随同一个shift增加：计划逆时针", fontsize=12)
    ax.grid(alpha=.18)
    ax = axes[1]
    for i in range(4):
        ax.scatter(shifts[i]*1000, point[i, 0]*1000, c=colors[i], s=65)
    ax.plot(shifts*1000, point[:, 0]*1000, color=".4", lw=1.6)
    ax.set(xlabel="共同 shift / mm", ylabel="同一P14的X / mm", title="该节点同时向接头（+X）送身")
    ax.grid(alpha=.22)
    ax.text(.05, .08, "P0=头，P21=尾\n箭头由P14的规划位置计算\n未用实际轨迹或不同杆替代", transform=ax.transAxes, fontsize=12, color="#4b5c6b")
    fig.text(.5, .025, "规划参考；实际滚动方向与驱动对应需在用户确认后验证。", ha="center", fontsize=11, color="#b14427")
    fig.subplots_adjust(left=.07, right=.97, top=.82, bottom=.15, wspace=.30)
    fig.savefig(OUT/"tail_to_head_ccw.png", dpi=180)
    plt.close(fig)
    return saved, dict(node_id=14, shifts_m=shifts.tolist(), points_m=point.tolist(),
                       unwrapped_theta_rad=theta.tolist(), ccw_increments_deg=np.degrees(np.diff(theta)).tolist(),
                       screen_right="-Y", screen_up="+Z", view_direction="+X", planned_not_actual=True)


if __name__ == "__main__":
    route = Route()
    check = geometry_check(route)
    if check["minimum_conservative_pipe_clearance_mm"] < -.05 or check["minimum_nonadjacent_rod_capsule_clearance_mm"] < -.05:
        raise ValueError(f"Reference geometry conflict: {check}")
    stages, direction = draw(route, pipe_surfaces())
    ss = np.linspace(-21*route.helix_step, route.final_shift+.08, 7001)
    points = route.sample(ss)
    np.savetxt(OUT/"global_curve.csv", np.column_stack([ss, points]), delimiter=",", header="station_m,x_m,y_m,z_m", comments="", fmt="%.10f")
    rows = []
    for stage in stages:
        for i, (point, station) in enumerate(zip(stage["points_m"], stage["stations_m"])):
            rows.append([stage["stage"], stage["shift_m"], i, station, *point])
    np.savetxt(OUT/"reference_windows.csv", rows, delimiter=",", header="stage,shared_shift_m,node_id,station_m,x_m,y_m,z_m", comments="", fmt="%.10f")
    metadata = dict(status="awaiting_user_curve_confirmation", planning_only=True, dynamics_steps=0,
                    source_exit_m=Q.tolist(), target_entry_m=E.tolist(), source_phi_deg=-30., target_phi_deg=-150.,
                    node_helix_radius_mm=R*1000, pitch_mm=PITCH*1000, source_phi_derivative="negative under positive curve station; tail-viewCCW",
                    bridge_length_mm=route.bridge_length*1000, helix_physical_chord_arc_step_mm=route.helix_step*1000,
                    initial_source_spatial_turns=21*route.helix_step*BETA/R/(2*np.pi),
                    final_shared_shift_mm=route.final_shift*1000, initial_tail_m=route.window(0)[0][-1].tolist(),
                    first_five_target_rods_shared_shift_mm=(route.bridge_length+5*route.helix_step)*1000,
                    stages=stages, same_node_direction=direction, geometry_checks=check)
    (OUT/"route_data.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT/"geometry_checks.json").write_text(json.dumps(check, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("stages", "same_node_direction")}, ensure_ascii=False, indent=2))
