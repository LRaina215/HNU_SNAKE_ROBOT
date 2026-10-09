"""Read-only model facts and continuum design drawings; no dynamics or controls."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco
from release_geometry import initial_curve, make_peel, INITIAL_METADATA

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False})
L, R, P = .0645, .0915, .180
BODY, FRONT, REAR, INTAKE = 21*L, 9*L, 12*L, 1.5*L
ALPHA = P/math.hypot(P, 2*math.pi*R)
BETA = math.sqrt(1-ALPHA**2)
GAMMA = math.asin(ALPHA)
RHO = INTAKE/GAMMA
E = np.array([1.1-R, .20, 1.])
A = E-np.array([0., RHO*(1-math.cos(GAMMA)), RHO*math.sin(GAMMA)])

def helix(v):
    v = np.atleast_1d(v)
    phi = math.pi-BETA*v/R
    return np.column_stack([1.1+R*np.cos(phi), .20+ALPHA*v, 1.+R*np.sin(phi)])

def receive_curve(v):
    v = np.atleast_1d(v)
    out = np.empty((len(v), 3))
    hanging, turn, wound = v < -INTAKE, (v >= -INTAKE) & (v < 0), v >= 0
    out[hanging] = A + (v[hanging]+INTAKE)[:, None]*np.array([0., 0., 1.])
    theta = (v[turn]+INTAKE)/RHO
    out[turn] = A+np.column_stack([np.zeros(len(theta)), RHO*(1-np.cos(theta)), RHO*np.sin(theta)])
    out[wound] = helix(v[wound])
    return out

def pipes(ax):
    a = np.linspace(0, 2*math.pi, 40)
    for direction, limits in [('x', (.52, 1.10)), ('y', (0., .68))]:
        s = np.linspace(*limits, 22)
        ss, aa = np.meshgrid(s, a)
        if direction == 'x':
            xx, yy, zz = ss, .0565*np.cos(aa), 1.+.0565*np.sin(aa)
        else:
            xx, yy, zz = 1.1+.0565*np.cos(aa), ss, 1.+.0565*np.sin(aa)
        ax.plot_surface(xx, yy, zz, color='#8ea3ab', alpha=.24, linewidth=0, shade=False)
    ax.plot([.52, 1.1, 1.1], [0, 0, .68], [1, 1, 1], '--', color='#8c9ba1', lw=.8)
    ax.plot([.72, 1.22], [.22, .22], [0, 0], color='#c9cfd2', lw=1)

COLORS = ['#176eab']*9+['#ee9c24']*2+['#8458ac']*10
def body(ax, sample, shift=0., alpha=1., labels=True):
    head = FRONT+shift
    for j in range(21):
        v = np.linspace(head-(j+1)*L, head-j*L, 24)
        c = sample(v)
        ax.plot(*c.T, color=COLORS[j], lw=3.2, alpha=alpha, solid_capstyle='round')
    centers = sample(head-(np.arange(21)+.5)*L)
    ax.scatter(*centers.T, s=9, color=COLORS, alpha=alpha, depthshade=False)
    if labels:
        for j in [9, 10]:
            c = centers[j]
            ax.text(c[0]-.015, c[1]-.022, c[2], 'M'+str(j), fontsize=8, color='#ae6300')
    tail, nose = sample([head-BODY, head])
    ax.scatter(*nose, s=27, color='#176eab', marker='>', depthshade=False, alpha=alpha)
    ax.scatter(*tail, s=15, color='#8458ac', depthshade=False, alpha=alpha)

fig = plt.figure(figsize=(17, 10.5), facecolor='white')
fig.suptitle('目标端承托 → 原管剥离 → 悬垂供给 → 连续收卷', fontsize=22, y=.982, color='#17354a')
fig.text(.5, .943, '完整动作组织示意｜连续体中心线与材料色标；不是实际关节链、碰撞验证或动力学结果', ha='center', fontsize=11, color='#596b77')
titles = ['① 候选预搭：目标 9 / 桥 6 / 原管 6 杆', '② 有序剥离：图示原管剩余 2 杆',
          '③ 悬垂就绪：目标 9 杆，共同 S = 0', '④ 第一杆收卷：S = 64.5 mm',
          '⑤ 第二杆收卷：S = 129 mm', '⑥ 循环 6 次后：S = 774 mm']
notes = ['近一整圈目标螺旋先承托；初态待验证', '释放4杆形成垂弯；完整过程见剥离图',
         '垂直供给 677 mm；尾端壳底约 199 mm', 'M9 进入螺旋；尾端上升 64.5 mm',
         'M9 / M10 留在螺旋；尾端累计上升 129 mm', '全 21 杆：同一螺旋，约 2.25 圈']
for k in range(6):
    ax = fig.add_subplot(2, 3, k+1, projection='3d')
    pipes(ax)
    if k == 0:
        body(ax, initial_curve)
    elif k == 1:
        body(ax, initial_curve, alpha=.18, labels=False)
        intermediate, _ = make_peel(2/3)
        body(ax, intermediate, labels=False)
    else:
        shift = [0., 0., 0., L, 2*L, REAR][k]
        body(ax, receive_curve, shift)
        if k in [3, 4]:
            base = receive_curve([FRONT-BODY])[0]
            now = receive_curve([FRONT+shift-BODY])[0]
            ax.plot([base[0]+.10, now[0]+.10], [base[1], now[1]], [base[2], now[2]], color='#8458ac', lw=1.6)
            ax.quiver(now[0]+.10, now[1], base[2], 0, 0, shift, color='#8458ac', arrow_length_ratio=.15, lw=1.5)
    ax.set_title(titles[k], fontsize=12, color='#17354a', pad=5)
    ax.text2D(.5, -.015, notes[k], transform=ax.transAxes, ha='center', fontsize=10, color='#3e5666')
    ax.set_xlim(.50, 1.30); ax.set_ylim(-.17, .72); ax.set_zlim(0, 1.22)
    ax.set_box_aspect([.80, .89, 1.22])
    ax.view_init(elev=18, azim=-54)
    ax.set_axis_off()
fig.subplots_adjust(left=.015, right=.985, top=.905, bottom=.105, wspace=.015, hspace=.19)
fig.text(.5, .062, '蓝色：起始目标前身 M0–M8   橙色：首轮接收 M9 / M10   紫色：其余后身（材料身份始终不变）', ha='center', fontsize=11)
fig.text(.5, .025, '③→④→⑤为首轮；随后同形收卷、S持续累加：每轮 +129 mm，前身转约 77°、轴向 +39 mm，后身上提 129 mm', ha='center', fontsize=12, color='#17354a')
fig.savefig(HERE/'cycle_overview.png', dpi=160)
fig.savefig(HERE/'cycle_overview.svg')
plt.close(fig)

# Side/axial view: exact nominal receiving curve and persistent material markers.
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
v = np.linspace(-REAR, BODY, 1800)
c = receive_curve(v)
axes[0].plot(c[:, 1]*1000, c[:, 2]*1000, '--', lw=1, color='#aab5bc')
for shift, color, label in [(0., '#c8c5d9', 'S = 0'), (2*L, '#8458ac', 'S = 129 mm')]:
    curve = receive_curve(np.linspace(FRONT+shift-BODY, FRONT+shift, 1000))
    axes[0].plot(curve[:, 1]*1000, curve[:, 2]*1000, color=color, lw=3, label=label)
axes[0].axhline(0, color='#848e94', lw=1)
axes[0].axhline(1000, color='#8ea3ab', lw=1)
axes[0].scatter([A[1]*1000, E[1]*1000], [A[2]*1000, E[2]*1000], c='#ee9c24')
axes[0].annotate('入口转弯：96.75 mm / 17.38°', xy=(A[1]*1000, A[2]*1000), xytext=(240, 650), arrowprops={'arrowstyle':'->'}, fontsize=10)
axes[0].annotate('尾端上升 129 mm', xy=(A[1]*1000, (A[2]-REAR+INTAKE+2*L)*1000), xytext=(240, 360), arrowprops={'arrowstyle':'->'}, fontsize=10)
axes[0].text(225, 85, '名义尾端壳底 199 mm\n未计摆动 / 挠曲 / 真链误差', fontsize=10)
axes[0].set(xlabel='目标管轴向 Y / mm', ylabel='世界高度 Z / mm', title='侧视：同一入口曲线，尾部供给持续缩短', xlim=(130, 700), ylim=(-20, 1170))
axes[0].legend(loc='upper right')
theta = np.linspace(0, 2*math.pi, 300)
axes[1].fill(.0565*1000*np.cos(theta), .0565*1000*np.sin(theta), color='#cdd6db')
axes[1].plot(R*1000*np.cos(theta), R*1000*np.sin(theta), '--', color='#8ea3ab')
for j in [0, 9, 10]:
    u = FRONT-(j+.5)*L+np.linspace(0, 2*L, 120)
    q = receive_curve(u)
    axes[1].plot((q[:, 0]-1.1)*1000, (q[:, 2]-1)*1000, color=COLORS[j], lw=3)
    axes[1].scatter((q[0, 0]-1.1)*1000, (q[0, 2]-1)*1000, marker='o', color=COLORS[j])
    axes[1].scatter((q[-1, 0]-1.1)*1000, (q[-1, 2]-1)*1000, marker='>', color=COLORS[j], s=55)
    axes[1].text((q[-1, 0]-1.1)*1000+3, (q[-1, 2]-1)*1000, 'M'+str(j), color=COLORS[j])
axes[1].text(-107, -128, '原有螺旋材料：转 77.1°、轴向 +38.5 mm\nM9 / M10 从入口卷入；圆点→三角', fontsize=10)
axes[1].set(xlabel='相对目标轴 X / mm', ylabel='相对目标轴 Z / mm', title='端视：同一批材料沿螺旋转动', xlim=(-125, 125), ylim=(-145, 125))
axes[1].set_aspect('equal')
fig.suptitle('收卷一整轮的材料运动预算（解析中心线设计）', fontsize=16)
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(HERE/'stroke_geometry.png', dpi=160)
plt.close(fig)

model = mujoco.MjModel.from_xml_path(str(REPO/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
names = ['head']+['link'+str(i) for i in range(1, 20)]+['tail']
masses = [float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]) for n in names]
facts = dict(status='design_only_not_dynamic_validation', model_read='MjModel only; no MjData, mj_step, or control writes',
    body_mass_kg=sum(masses), rod_mass_kg=masses, gravity_m_s2=model.opt.gravity.tolist(),
    actuator_force_range_nm=model.actuator_forcerange.tolist(),
    rod_length_mm=L*1000, body_length_mm=BODY*1000, pipe_radius_mm=56.5,
    nominal_body_radius_mm=29., design_helix_radius_mm=R*1000, pitch_mm=P*1000,
    axial_fraction=ALPHA, angular_rate_rad_per_m=-BETA/R,
    initial_target_rods=9, initial_bridge_rods=6, initial_source_rods=6,
    intake_arc_length_mm=INTAKE*1000, intake_angle_deg=math.degrees(GAMMA), intake_radius_mm=RHO*1000,
    entry_point_mm=(E*1000).tolist(), intake_start_mm=(A*1000).tolist(),
    hanging_length_mm=(REAR-INTAKE)*1000, tail_endpoint_height_mm=(A[2]-REAR+INTAKE)*1000,
    nominal_tail_shell_bottom_mm=(A[2]-REAR+INTAKE-.029)*1000,
    hanging_supply_mass_kg=sum(masses[9:]), hanging_supply_weight_n=sum(masses[9:])*9.81,
    weight_times_radius_nm=sum(masses[9:])*9.81*R,
    torque_scale_caveat='weight times radius is not any joint torque, grip capacity, or a feasibility proof',
    stroke_mm=2*L*1000, stroke_rotation_deg=math.degrees(2*L*BETA/R), stroke_axial_mm=2*L*ALPHA*1000,
    full_feed_mm=REAR*1000, final_turns=BODY*BETA/(2*math.pi*R), final_axis_span_mm=BODY*ALPHA*1000,
    continuous_integral_curvature_bound_deg=math.degrees(2*L*BETA**2/R),
    initial_bridge_design_handle_mm=INITIAL_METADATA['handle_mm'],
    initial_bridge_arclength_mm=INITIAL_METADATA['connector_length_mm']+INTAKE*1000,
    limitations=['release continuum keyframes constructed, physically admissible sweep not established', 'initial real-chain placement not solved', 'rigid shell/self clearance not tested',
                 'traction/holding/lifting not tested', 'line-arc-helix is C1, curvature blends still required'])
(HERE/'design_budget.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:facts[k] for k in ['hanging_supply_weight_n','nominal_tail_shell_bottom_mm','stroke_rotation_deg','stroke_axial_mm','final_turns','continuous_integral_curvature_bound_deg','initial_bridge_arclength_mm']}, indent=2))
