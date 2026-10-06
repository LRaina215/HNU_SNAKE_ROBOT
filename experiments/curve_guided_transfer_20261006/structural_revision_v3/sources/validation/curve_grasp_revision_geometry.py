"""One offline constrained discretization; no physical stepping or parameter scan."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import math
import numpy as np
import mujoco
from scipy.optimize import minimize,OptimizeResult
from scipy.spatial.transform import Rotation,Slerp
from scipy.spatial import ConvexHull
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.curve_guided_transfer import FiveSegmentCurve, CurveParameters
from validation.curve_guided_geometry import points, placement, MODEL, CORNER

BASE = ROOT / 'experiments/curve_guided_transfer_20261006/structural_revision_v3'
OLD = BASE.parent
PARAMETERS = CurveParameters(radius_mm=95., initial_target_links=7.)


def write(name, data):
    (BASE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def local_vertices(model, geom):
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half = model.geom_size[geom, :2]
        angle = np.arange(32)*2*math.pi/32
        return np.vstack([np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.full(32,z))) for z in [-half,half]])
    mesh = model.mesh(int(model.geom_dataid[geom]))
    vertices=model.mesh_vert[int(mesh.vertadr[0]):int(mesh.vertadr[0]+mesh.vertnum[0])].copy()
    return vertices[ConvexHull(vertices).vertices]


def projected_gap(vertices, side):
    xy = (vertices*1000-CORNER)[:, [1,2] if side == 0 else [0,2]]
    # Andrew monotone chain: same convex polygon, no per-query Qhull temporary
    # stderr file creation on Windows. Static 3D hulls still use SciPy once.
    ordered=sorted(set(map(tuple,xy.tolist())))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[]
    for p in ordered:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(ordered):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    polygon=np.array(lower[:-1]+upper[:-1])
    if all(cross(a,b,(0.,0.))>=0 for a,b in zip(polygon,np.roll(polygon,-1,axis=0))):
        return -56.5
    a, b = polygon, np.roll(polygon,-1,axis=0)
    edge = b-a
    t = np.clip(-np.sum(a*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
    return float(np.linalg.norm(a+t[:,None]*edge,axis=1).min()-56.5)


class DiscreteFit:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.curve = FiveSegmentCurve(PARAMETERS)
        self.qadr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.robot = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] >= self.m.body('head').id]
        self.pipes = [self.m.geom(n).id for n in ['pipe_straight_01', 'pipe_sharp_in_02', 'pipe_sharp_out_03', 'pipe_straight_04']]
        self.vertices = []
        for g in self.robot:
            self.vertices.append(local_vertices(self.m,g))
        self.last = None

    def set(self, x):
        if self.last is not None and np.array_equal(self.last, x):
            return
        self.last = x.copy()
        self.d.qpos[:3] = x[:3] / 1000
        q = Rotation.from_rotvec(x[3:6]).as_quat()
        self.d.qpos[3:7] = q[[3, 0, 1, 2]]
        self.d.qpos[self.qadr] = x[6:]
        mujoco.mj_kinematics(self.m, self.d)

    def gaps(self, x):
        self.set(x)
        # Exact finite-L / self collision is checked after fitting and at all
        # interpolation validation postures, not during every numerical derivative.
        return np.array([self.radial_gap(module,side) for module in range(21) for side in [0,1]])

    def radial_gap(self,module,side):
        g=self.robot[module]
        vertices=self.vertices[module] @ self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g]
        return projected_gap(vertices,side)

    def vertex_gap(self, module, side):
        g = self.robot[module]
        vertices = self.vertices[module] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
        return projected_gap(vertices,side)

    def fit(self, feed, previous=None):
        placement(self.m, self.d, self.curve, feed, fit=True)
        x0 = np.r_[self.d.qpos[:3]*1000, Rotation.from_quat(self.d.qpos[3:7][[1,2,3,0]]).as_rotvec(), self.d.qpos[self.qadr]]
        desired = self.curve.sample(self.curve.lambda0+feed-np.arange(22)*64.5)+CORNER
        nominal = self.curve.joint_angles(feed)
        ell = self.curve.material_positions(feed)
        target = np.flatnonzero(ell > self.curve.bounds[-1]+64.5)
        source = np.flatnonzero(ell < -64.5)
        grips = [(int(target[1]), 1), (int(target[-2]), 1), (int(source[1]), 0), (int(source[-2]), 0)]
        if previous is not None:
            nominal_now=x0.copy()
            x0 = previous + (nominal_now-self.previous_nominal)
            rotation = Rotation.from_rotvec(nominal_now[3:6])*Rotation.from_rotvec(self.previous_nominal[3:6]).inv()*Rotation.from_rotvec(previous[3:6])
            x0[3:6]=rotation.as_rotvec()
        initial_nominal = x0.copy() if previous is None else nominal_now
        self.last = None

        def objective(x):
            self.set(x)
            error = (points(self.m, self.d)-desired)/10
            closeness = np.array([self.vertex_gap(module, side)-.2 for module, side in grips])/2
            joint_error = (x[6:]-nominal)/math.radians(20)
            return float(np.mean(error**2) + np.mean(closeness**2) + .03*np.mean(joint_error**2))

        bounds = [(x0[i]-100, x0[i]+100) for i in range(3)]
        bounds += [(x0[i]-.8, x0[i]+.8) for i in range(3, 6)]
        bounds += [(-math.pi/2, math.pi/2)]*20
        # Numerical units only: preserve the same geometry/objective/constraints.
        units = np.r_[np.full(3,10.),np.full(23,.1)]
        unpack = lambda z:x0+units*z
        scaled_bounds = [((lo-x0[i])/units[i],(hi-x0[i])/units[i]) for i,(lo,hi) in enumerate(bounds)]
        iterations=[0]
        feasible=[None]
        class GeometryFeasible(Exception):
            pass
        def progress(z):
            iterations[0]+=1
            if iterations[0]%30==0:
                print(json.dumps(dict(feed_mm=float(feed),offline_iteration=iterations[0],physics_steps=0)),flush=True)
            if iterations[0]<10:
                return
            x=unpack(z);self.set(x)
            rms=float(np.sqrt(np.mean(np.sum((points(self.m,self.d)-desired)**2,axis=1))))
            if rms<=25 and self.gaps(x).min()>=0 and all(self.radial_gap(i,s)<=1. for i,s in grips):
                mujoco.mj_forward(self.m,self.d)
                if not any(c.dist<0 for c in self.d.contact):
                    feasible[0]=x.copy()
                    raise GeometryFeasible()
        try:
            result = minimize(lambda z:objective(unpack(z)), np.zeros(26), method='SLSQP', bounds=scaled_bounds,
                              constraints=[dict(type='ineq', fun=lambda z:self.gaps(unpack(z))-.05)],
                              callback=progress,options=dict(maxiter=180, ftol=1e-6, eps=1e-5))
            result.x = unpack(result.x)
        except GeometryFeasible:
            result=OptimizeResult(x=feasible[0],success=False,message='stopped on checked geometric admission, not an optimality claim',
                fun=objective(feasible[0]),nit=iterations[0])
        self.set(result.x)
        mujoco.mj_forward(self.m, self.d)
        gaps = self.gaps(result.x)
        exact_grip_gaps = [self.radial_gap(module,side) for module,side in grips]
        row = dict(feed_mm=float(feed), optimizer_success=bool(result.success), optimizer_message=str(result.message),
                   iterations=int(result.nit), objective=float(result.fun), minimum_gap_mm=float(gaps.min()),
                   grip_modules=[dict(module=module, side='target' if side else 'source', gap_mm=gap) for (module,side),gap in zip(grips,exact_grip_gaps)],
                   maximum_angle_deg=float(np.degrees(abs(result.x[6:])).max()),
                   fk_reference_rms_point_mm=float(np.sqrt(np.mean(np.sum((points(self.m,self.d)-desired)**2,axis=1)))),
                   maximum_fk_reference_point_mm=float(np.linalg.norm(points(self.m,self.d)-desired,axis=1).max()),
                   qpos=self.d.qpos.tolist(), actual_points_mm=points(self.m,self.d).tolist(), reference_points_mm=desired.tolist(),
                   exact_collision_contacts=[dict(geom1=self.m.geom(c.geom1).name,geom2=self.m.geom(c.geom2).name,dist_mm=float(c.dist*1000)) for c in self.d.contact])
        self.previous_nominal = initial_nominal
        return result.x, row


def diagnose():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] >= model.body('head').id]
    output = {}
    for case in ['V1_C0', 'V2_C0']:
        controls = json.loads((OLD/case/'control.json').read_text(encoding='utf8'))
        delta = np.degrees(np.array([r['compliance_delta_rad'] for r in controls]))
        with np.load(OLD/case/'trace.npz') as trace:
            data.qpos[:] = trace['qpos'][0]
            mujoco.mj_forward(model,data)
            # Surface-vertex radial clearance is a diagnostic proxy, not signed mesh distance.
            radial = []
            for g in robot:
                v = local_vertices(model,g)
                v = v @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
                p = v*1000-CORNER
                radial.append([float(np.linalg.norm(p[:,dims],axis=1).min()-56.5) for dims in [[1,2],[0,2]]])
            output[case] = dict(maximum_ad_delta_deg=float(abs(delta).max()), clamped_joint_updates=int(np.count_nonzero(abs(delta)>.999999)),
                commands=len(controls), initial_contact_modules=[np.flatnonzero(trace['contact_force_N'][0,s]>.1).tolist() for s in [0,1]],
                initial_surface_vertex_radial_gap_mm=radial, maximum_joint_change_deg=float(np.degrees(abs(trace['q']-trace['q'][0])).max()))
    write('previous_evidence.json', output)


def main():
    BASE.mkdir(exist_ok=True)
    if (BASE/'geometry_result.json').exists():
        raise FileExistsError('Preserve registered revision geometry')
    diagnose()
    fit = DiscreteFit()
    # One fixed curve and a fixed material lattice. This is not a radius/gain search.
    feeds = np.arange(0, 161.25+.001, 16.125)
    rows, xs = [], []
    previous = None
    partial=BASE/'geometry_result_partial.json'
    if partial.exists():
        rows=json.loads(partial.read_text(encoding='utf8'))['rows']
        with np.load(BASE/'fitted_reference_partial.npz') as archive:
            xs=list(archive['states'])
        previous=xs[-1].copy()
        placement(fit.m,fit.d,fit.curve,rows[-1]['feed_mm'],fit=True)
        fit.previous_nominal=np.r_[fit.d.qpos[:3]*1000,Rotation.from_quat(fit.d.qpos[3:7][[1,2,3,0]]).as_rotvec(),fit.d.qpos[fit.qadr]]
    for feed in feeds[len(rows):]:
        previous, row = fit.fit(feed, previous)
        rows.append(row); xs.append(previous.copy())
        write('offline_progress.json',dict(physics_steps=0, rows=rows))
        print(json.dumps({k:row[k] for k in ['feed_mm','optimizer_success','minimum_gap_mm','grip_modules','maximum_angle_deg','fk_reference_rms_point_mm']},ensure_ascii=False),flush=True)
        if row['minimum_gap_mm'] < 0 or row['fk_reference_rms_point_mm'] > 25 or any(c['dist_mm']<0 for c in row['exact_collision_contacts']):
            break
    ready = len(rows)==len(feeds) and all(r['minimum_gap_mm'] >= 0 and r['maximum_angle_deg'] < 90 and
        not any(c['dist_mm'] < 0 for c in r['exact_collision_contacts']) and all(g['gap_mm'] <= 1. for g in r['grip_modules']) for r in rows)
    xs = np.array(xs)
    interpolation = []
    maximum_step = None
    if ready:
        interpolator = PchipInterpolator(feeds,xs,axis=0)
        orientations = Slerp(feeds,Rotation.from_rotvec(xs[:,3:6]))
        for feed in np.arange(0,161.25+.001,8.0625):
            x=interpolator(feed);x[3:6]=orientations(feed).as_rotvec();fit.set(x);mujoco.mj_forward(fit.m,fit.d)
            interpolation.append(dict(feed_mm=float(feed),minimum_gap_mm=float(fit.gaps(x).min()),maximum_angle_deg=float(np.degrees(abs(x[6:])).max()),
                actual_minimum_contact_mm=min((float(c.dist*1000) for c in fit.d.contact),default=0.)))
        ready = all(r['minimum_gap_mm'] >= -.05 and r['actual_minimum_contact_mm'] >= 0 for r in interpolation)
        q = interpolator(np.arange(0,161.25+.001,.25))[:,6:]
        maximum_step=float(np.degrees(abs(np.diff(q,axis=0))).max())
        ready &= maximum_step <= .5
    result=dict(design_status='READY_FOR_MINIMAL_SIM' if ready else 'NOT_READY', physics_steps=0,
        fixed_parameters=vars(PARAMETERS), length_mm=fit.curve.length, transition_length_mm=float(fit.curve.bounds[-1]),
        solver_variable_units='10 mm translation / 0.1 rad rotation and joint angle; same objective and constraints as numerical_attempt0',
        rows=rows, interpolation_checks=interpolation, maximum_50ms_nominal_step_deg=maximum_step,
        limitations=['Every offline pose has an artificial free-base placement; no grip-force or movement claim.',
                     'Contact-near fits do not prove static equilibrium; exact contacts/world poses remain offline only.',
                     'Infinite-cylinder projected-hull gaps are conservative offline geometry; finite L contacts are checked with mj_collision.',
                     'mj_geomDistance artifacts in numerical attempts 0/1 were excluded, not treated as actual penetration.'])
    write('geometry_result.json',result)
    np.savez_compressed(BASE/'fitted_reference.npz', feed_mm=feeds[:len(rows)], states=xs)
    if ready:
        qpos=np.array(rows[0]['qpos'])
        np.savez_compressed(BASE/'initial_pose.npz',qpos=qpos,qvel=np.zeros(fit.m.nv),target=qpos[fit.qadr])
    fig=plt.figure(figsize=(12,7))
    for i,index in enumerate(np.linspace(0,len(rows)-1,min(6,len(rows)),dtype=int)):
        row=rows[index]; ax=fig.add_subplot(2,3,i+1,projection='3d')
        ax.plot(*np.array(row['reference_points_mm']).T,'--',color='grey')
        ax.plot(*np.array(row['actual_points_mm']).T,'o-',markersize=3)
        ax.plot([500,1100],[0,0],[1000,1000],color='black',linewidth=5)
        ax.plot([1100,1100],[0,600],[1000,1000],color='black',linewidth=5)
        ax.set_title(f'feed={row["feed_mm"]:g} mm; offline placement')
        ax.set_xlim(650,1300);ax.set_ylim(-130,500);ax.set_zlim(870,1170);ax.view_init(25,-65)
    fig.suptitle('V3 fixed R95 curve with finite-shell constrained discretization. No dynamics or support claim.')
    fig.tight_layout();fig.savefig(BASE/'key_postures.png',dpi=150);plt.close(fig)
    print(result['design_status'],flush=True)


if __name__=='__main__':
    main()
