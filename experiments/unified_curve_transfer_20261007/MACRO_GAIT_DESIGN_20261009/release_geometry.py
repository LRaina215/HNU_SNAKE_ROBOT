"""Fixed-length release keyframes. Pure geometry, no robot controller or physics."""
from pathlib import Path
import math
import json
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
L, R, P = .0645, .0915, .180
ALPHA = P/math.hypot(P, 2*math.pi*R)
BETA = math.sqrt(1-ALPHA**2)
INTAKE = 1.5*L
GAMMA = math.asin(ALPHA)
RHO = INTAKE/GAMMA
E = np.array([1.1-R, .20, 1.])
A = E-np.array([0., RHO*(1-math.cos(GAMMA)), RHO*math.sin(GAMMA)])
Q = np.array([.88, R, 1.])
UP = np.array([0., 0., 1.])
GRID = np.linspace(0., 1., 2401)

def source_curve(u):
    u = np.atleast_1d(u)
    phi = BETA*u/R
    return np.column_stack([Q[0]+ALPHA*u, R*np.cos(phi), 1.+R*np.sin(phi)])

def source_tangent(u):
    phi = BETA*u/R
    return np.array([ALPHA, -BETA*math.sin(phi), BETA*math.cos(phi)])

def receiving(v):
    v = np.atleast_1d(v)
    out = np.empty((len(v), 3))
    low, arc, high = v < -INTAKE, (v >= -INTAKE) & (v < 0), v >= 0
    out[low] = A+(v[low]+INTAKE)[:, None]*UP
    theta = (v[arc]+INTAKE)/RHO
    out[arc] = A+np.column_stack([np.zeros(len(theta)), RHO*(1-np.cos(theta)), RHO*np.sin(theta)])
    theta = math.pi-BETA*v[high]/R
    out[high] = np.column_stack([1.1+R*np.cos(theta), .20+ALPHA*v[high], 1.+R*np.sin(theta)])
    return out

def connector(start, tangent, length):
    """Hermite centerline with prescribed end tangents, arc-length reparameterized."""
    chord = np.linalg.norm(A-start)
    if chord > length+1e-8:
        raise ValueError('Connector chord exceeds available body length')
    if length-chord < 1e-8:
        # Final vertical route has zero slack, and both tangents are vertical.
        points = start+GRID[:, None]*(A-start)
        return points, GRID*length, 0.
    def raw(h):
        t = GRID[:, None]
        return ((2*t**3-3*t**2+1)*start+(t**3-2*t**2+t)*h*tangent
                +(-2*t**3+3*t**2)*A+(t**3-t**2)*h*UP)
    def arc(h):
        return np.linalg.norm(np.diff(raw(h), axis=0), axis=1).sum()
    h = brentq(lambda h: arc(h)-length, .000001, 4.)
    points = raw(h)
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, cumulative, h

def make_peel(q):
    # q releases source material from its outlet backwards toward the tail.
    cut = -6*L-q*6*L
    source_u = cut+6*L
    start = source_curve([source_u])[0]
    free_length = -cut
    points, cumulative, handle = connector(start, source_tangent(source_u), free_length-INTAKE)
    def curve(v):
        v = np.atleast_1d(v)
        out = np.empty((len(v), 3))
        kept, span, entry = v < cut, (v >= cut) & (v < -INTAKE), v >= -INTAKE
        out[kept] = source_curve(v[kept]+6*L)
        out[span] = np.column_stack([np.interp(v[span]-cut, cumulative, points[:, k]) for k in range(3)])
        out[entry] = receiving(v[entry])
        return out
    metadata = dict(q=q, source_rods_retained=6*(1-q), released_source_rods=6*q,
        fixed_target_rods=9, free_rear_length_mm=free_length*1000,
        connector_length_mm=float(cumulative[-1]*1000), intake_length_mm=INTAKE*1000,
        retained_source_length_mm=6*L*(1-q)*1000, cut_point_mm=(start*1000).tolist(),
        minimum_connector_height_mm=float(points[:,2].min()*1000), handle_mm=handle*1000)
    return curve, metadata

def make_unfold(q):
    # Tail first exits radially and goes around the source underside. A diagonal
    # from the old tail straight to the side gate would cut across the source pipe.
    start0 = source_curve([-6*L])[0]
    end = A-(12*L-INTAKE)*UP
    gate = np.array([.91, .15, .48])
    theta0 = math.atan2(start0[2]-1.,start0[1])
    smooth = lambda z: z*z*(3-2*z)
    if q <= .15:
        radius = R+smooth(q/.15)*(.130-R)
        start = np.array([start0[0],radius*math.cos(theta0),1.+radius*math.sin(theta0)])
    elif q <= .40:
        blend = smooth((q-.15)/.25)
        theta = theta0+blend*(1.5*math.pi-theta0)
        start = np.array([start0[0]+blend*(.80-start0[0]),.130*math.cos(theta),1.+.130*math.sin(theta)])
    elif q <= .70:
        blend = smooth((q-.40)/.30)
        start = np.array([.80,0.,.87])+blend*(gate-np.array([.80,0.,.87]))
    else:
        blend = smooth((q-.70)/.30)
        start = gate+blend*(end-gate)
    tangent0 = source_tangent(-6*L)
    tangent = (1-q)*tangent0+q*UP
    tangent /= np.linalg.norm(tangent)
    points, cumulative, handle = connector(start, tangent, 12*L-INTAKE)
    bow = 0.
    if .40 < q < 1.:
        # Keep the free-tail bend broad: trade the long Hermite handles for a
        # lateral bow below the pipes, instead of making a tiny fold at the tail.
        blend = smooth(min(1.,(q-.40)/.15))
        handle += blend*(min(handle,np.linalg.norm(A-start))-handle)
        t = GRID[:,None]
        base = ((2*t**3-3*t**2+1)*start+(t**3-2*t**2+t)*handle*tangent
                +(-2*t**3+3*t**2)*A+(t**3-t**2)*handle*UP)
        weight = 16*t*t*(1-t)*(1-t)
        def bent(b):
            return base+weight*b*np.array([-1.,0.,0.])
        def arc(b):
            return np.linalg.norm(np.diff(bent(b),axis=0),axis=1).sum()
        if arc(0.) < 12*L-INTAKE-1e-10:
            bow = brentq(lambda b:arc(b)-(12*L-INTAKE),0.,2.)
        points = bent(bow)
        cumulative = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    def curve(v):
        v = np.atleast_1d(v)
        out = receiving(v)
        span = v < -INTAKE
        out[span] = np.column_stack([np.interp(v[span]+12*L, cumulative, points[:, k]) for k in range(3)])
        return out
    metadata = dict(q=q, source_rods_retained=0, fixed_target_rods=9,
        free_rear_length_mm=12*L*1000, connector_length_mm=float(cumulative[-1]*1000),
        intake_length_mm=INTAKE*1000, tail_point_mm=(start*1000).tolist(),
        minimum_connector_height_mm=float(points[:,2].min()*1000), handle_mm=handle*1000,
        lateral_bow_mm=bow*1000)
    return curve, metadata

initial_curve, INITIAL_METADATA = make_peel(0.)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(), 'axes.unicode_minus':False})
    colors = ['#176eab']*9+['#ee9c24']*2+['#8458ac']*10
    states = [make_peel(0), make_peel(1/3), make_peel(2/3), make_peel(1), make_unfold(.70), make_unfold(1)]
    titles = ['① 原管6杆 / 自由桥6杆', '② 原管4杆 / 自由后身8杆', '③ 原管2杆 / 自由后身10杆',
              '④ 原管0杆 / 自由后身12杆', '⑤ 尾端向目标侧下方转移', '⑥ 垂弯展开为直供给段']
    notes = ['源出口在靠近目标的侧面；开始按序剥离', '两杆从出口脱开，释放长度加到管外垂弯',
             '再释放两杆；剩余源侧后端保持原位置', '尾端最后解除原管；此时仍有垂弯',
             '尾端先绕原管外侧/底部，再移至侧下方', '12杆 = 1.5杆入口弧 + 10.5杆直供给']
    fig = plt.figure(figsize=(17, 10.7), facecolor='white')
    fig.suptitle('原管退出的宏观过程：包绕长度 → 管外垂弯 → 悬垂供给', fontsize=20, y=.978, color='#17354a')
    fig.text(.5, .943, '所有形态：目标前9杆不变、共同S=0、全身1354.5 mm；已构造等长C1中心线，未验证壳体/铰链/承载', ha='center', fontsize=11, color='#596b77')
    for k, ((sample, metadata), title, note) in enumerate(zip(states, titles, notes)):
        ax = fig.add_subplot(2, 3, k+1, projection='3d')
        a = np.linspace(0, 2*math.pi, 32)
        for direction, limits in [('x', (.52, 1.1)), ('y',(0., .65))]:
            ss, aa = np.meshgrid(np.linspace(*limits, 20), a)
            if direction == 'x': xx,yy,zz = ss,.0565*np.cos(aa),1+.0565*np.sin(aa)
            else: xx,yy,zz = 1.1+.0565*np.cos(aa),ss,1+.0565*np.sin(aa)
            ax.plot_surface(xx,yy,zz,color='#8ea3ab',alpha=.24,linewidth=0,shade=False)
        for j in range(21):
            v = np.linspace(9*L-(j+1)*L,9*L-j*L,25)
            c = sample(v)
            ax.plot(*c.T,color=colors[j],lw=3.5,solid_capstyle='round')
        centers = sample(9*L-(np.arange(21)+.5)*L)
        ax.scatter(*centers.T,s=9,color=colors,depthshade=False)
        ax.scatter(*sample([-12*L])[0],s=25,color='#8458ac',depthshade=False)
        if 'cut_point_mm' in metadata:
            cut = np.array(metadata['cut_point_mm'])/1000
            ax.scatter(*cut,s=35,color='#df4444',depthshade=False)
            ax.text(cut[0]-.06,cut[1]-.07,cut[2]+.045,'剥离边界',fontsize=8,color='#be3838')
        ax.plot([.62,1.24],[.22,.22],[0,0],color='#c9cfd2')
        ax.set_title(title,fontsize=12,pad=5,color='#17354a')
        ax.text2D(.5,-.025,note,transform=ax.transAxes,ha='center',fontsize=10,color='#3e5666')
        ax.set_xlim(.5,1.32);ax.set_ylim(-.3,.70);ax.set_zlim(0,1.30)
        ax.set_box_aspect([.82,1.,1.30]);ax.view_init(elev=18,azim=-54);ax.set_axis_off()
    fig.subplots_adjust(left=.01,right=.99,top=.905,bottom=.105,wspace=.015,hspace=.18)
    fig.text(.5,.06,'长度账：原管保留 387→258→129→0 mm；自由后身 387→516→645→774 mm（包含固定96.75 mm入口弧）',ha='center',fontsize=11)
    fig.text(.5,.025,'④→⑤→⑥不增加供给长度，也不前推shift；全源释放后才把垂弯展开，再开始共同shift收卷',ha='center',fontsize=12,color='#17354a')
    fig.savefig(HERE/'release_sequence.png',dpi=160)
    fig.savefig(HERE/'release_sequence.svg')
    plt.close(fig)
    rows = [m for _,m in states]
    (HERE/'release_budget.json').write_text(json.dumps(dict(status='constructed_continuum_centerline_only',
        construction='arc-length parameterized cubic Hermite; late unfolding adds lateral quartic bow. Target, retained source and intake arc stay fixed.',
        keyframes=rows, limitations=['no rigid-shell clearance verification','no joint reconstruction','no dynamics',
        'intermediate continuum curves not proven physically admissible']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Necessary diagram checks only: continuity and a sampled axis envelope.
    # These are not collision tests of the actual articulated robot or elbow.
    samples = np.linspace(-12*L,9*L,1801)
    checks=[]
    for family,make in [('peel',make_peel),('unfold',make_unfold)]:
        for q in np.linspace(0,1,21):
            sample,meta=make(float(q)); c=sample(samples)
            source_axis=np.column_stack([np.clip(c[:,0],-.2,.95),np.zeros(len(c)),np.ones(len(c))])
            target_axis=np.column_stack([np.full(len(c),1.1),np.clip(c[:,1],.15,1.2),np.ones(len(c))])
            checks.append(dict(family=family,q=float(q),source_axis_min_mm=float(np.linalg.norm(c-source_axis,axis=1).min()*1000),
                target_axis_min_mm=float(np.linalg.norm(c-target_axis,axis=1).min()*1000),minimum_centerline_z_mm=float(c[:,2].min()*1000)))
    junction_error=np.max(np.linalg.norm(make_peel(1)[0](samples)-make_unfold(0)[0](samples),axis=1))*1000
    final_error=np.max(np.linalg.norm(make_unfold(1)[0](samples)-receiving(samples),axis=1))*1000
    result=dict(meaning='Sampled continuum centerline relative to the two straight pipe axes only; no elbow mesh, rod chords, rigid shells, joints or dynamics.',
        phase_count=42,points_per_phase=len(samples),source_axis_min_mm=min(x['source_axis_min_mm'] for x in checks),
        target_axis_min_mm=min(x['target_axis_min_mm'] for x in checks),minimum_centerline_z_mm=min(x['minimum_centerline_z_mm'] for x in checks),
        peel_to_unfold_position_error_mm=float(junction_error),unfold_to_receive_position_error_mm=float(final_error),phases=checks)
    (HERE/'release_geometry_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='phases'},indent=2))
