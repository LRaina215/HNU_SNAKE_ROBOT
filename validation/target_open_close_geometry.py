"""One R95->130->95 target-side action, offline geometry only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation')]
import json
import math
import time
import shutil
import csv
import numpy as np
import mujoco
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from scipy.integrate import cumulative_trapezoid
from curve_local_path import LocalPathFit
from curve_guided_geometry import points, CORNER
from snake_control.local_material_reference import BoundaryMaterialCurve
from snake_control.target_open_close_reference import TargetOpenCloseReference

PARENT = ROOT/'experiments/curve_guided_transfer_20261006'
OLD = PARENT/'stage_continuation_20261006'
BASE = PARENT/'target_open_close_20261006'
ORIGINAL = OLD/'local_path_boundary_speed3/reference_ready.npz'


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def expanded_ideal(curve, ell, amplitude):
    """R'=dR/dsigma; phi'=sqrt(1-R'^2)/sqrt(R^2+h^2).

    Fixed pitch per radian h, unit arclength, unchanged target entry tangent.
    Radius changes only after target-body segment entry, tapered over 2 links.
    """
    p = curve.sample(ell).copy()
    sigma = np.asarray(ell)-curve.bounds[-1]
    if amplitude == 0 or not np.any(sigma > 0):
        return p
    grid = np.unique(np.r_[np.arange(0., float(sigma.max())+.251, .25), sigma[sigma>0]])
    L = 2*64.5
    u = np.clip(grid/L, 0, 1)
    radius = 95.+amplitude*u*u*(3-2*u)
    dr = amplitude*6*u*(1-u)/L
    pitch_per_radian = 220/(2*np.pi)
    if max(abs(dr)) >= 1:
        raise ValueError('Variable-radius profile exceeds available arclength')
    phase = cumulative_trapezoid(np.sqrt(1-dr*dr)/np.sqrt(radius*radius+pitch_per_radian**2), grid, initial=0.)
    ids = np.flatnonzero(sigma > 0)
    phi = np.interp(sigma[ids], grid, phase)
    rr = np.interp(sigma[ids], grid, radius)
    p[ids, 0] = rr*np.sin(phi)
    p[ids, 1] = curve.E[1]+pitch_per_radian*phi
    p[ids, 2] = rr*np.cos(phi)
    return p


class OpeningFit:
    def __init__(self):
        self.fit = LocalPathFit()
        self.old = BoundaryMaterialCurve(ORIGINAL)
        self.anchor = self.fit.m.geom_bodyid[self.fit.robot[10]]

    def begin(self, feed, amplitude):
        self.oldx = self.old.state(feed)
        self.fit.set(self.oldx)
        self.oldpoints = points(self.fit.m, self.fit.d).copy()
        self.anchor_pos = self.fit.d.xpos[self.anchor].copy()
        self.anchor_rot = self.fit.d.xmat[self.anchor].reshape(3,3).copy()
        ell = self.old.lambda0+feed-np.arange(22)*64.5
        ideal = self.old.sample(ell)
        self.desired = self.oldpoints + expanded_ideal(self.old, ell, amplitude)-ideal

    def unpack(self, q):
        x = self.oldx.copy()
        x[6:16] = q
        self.fit.set(x)
        rotate = self.anchor_rot @ self.fit.d.xmat[self.anchor].reshape(3,3).T
        anchor_current = self.fit.d.xpos[self.anchor].copy()
        x[:3] = (self.anchor_pos+rotate@(x[:3]/1000-anchor_current))*1000
        x[3:6] = (Rotation.from_matrix(rotate)*Rotation.from_rotvec(x[3:6])).as_rotvec()
        self.fit.set(x)
        return x

    def solve(self, action, previous):
        feed = TargetOpenCloseReference.material_feed(action)
        amplitude = TargetOpenCloseReference.opening(action)
        self.begin(feed, amplitude)
        if amplitude == 0:
            return self.oldx.copy(), dict(action_mm=float(action), feed_mm=feed, opening_mm=0., exact_original=True)
        seed = previous[6:16]
        def residual(q):
            x = self.unpack(q)
            error = (points(self.fit.m, self.fit.d)[:10]-self.desired[:10])/5.
            return np.r_[error.ravel(), .03*(q-seed)/np.radians(10.)]
        ls = least_squares(residual, np.clip(seed, -np.radians(89), np.radians(89)),
                           bounds=(-np.radians(89), np.radians(89)), max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        def constraints(q):
            x = self.unpack(q)
            # The downstream tail is fixed exactly, including original .0485mm
            # conservative projection deficit. It cannot be altered by front
            # variables; actual finite collisions are still checked everywhere.
            return self.fit.gaps(x)[:20]-.05
        initial = self.unpack(ls.x)
        if constraints(ls.x).min() < -.001:
            result = minimize(lambda q:float(residual(q)@residual(q)), ls.x,
                method='SLSQP', bounds=[(-np.radians(89), np.radians(89))]*10,
                constraints=[dict(type='ineq', fun=constraints)],
                options=dict(maxiter=100, ftol=1e-8, eps=1e-5))
            x = self.unpack(result.x)
            solver = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit))
        else:
            x = initial
            solver = dict(success=bool(ls.success), message=str(ls.message), iterations=int(ls.nfev))
        mujoco.mj_forward(self.fit.m, self.fit.d)
        p = points(self.fit.m, self.fit.d)
        collisions = [dict(geom1=self.fit.m.geom(c.geom1).name, geom2=self.fit.m.geom(c.geom2).name, distance_mm=float(c.dist*1000)) for c in self.fit.d.contact if c.dist < 0]
        rms = float(np.sqrt(np.mean(np.sum((p[:10]-self.desired[:10])**2, axis=1))))
        row = dict(action_mm=float(action), feed_mm=feed, opening_mm=amplitude, solver=solver,
            desired_front_rms_mm=rms, minimum_conservative_gap_mm=float(self.fit.gaps(x).min()),
            maximum_joint_deg=float(np.degrees(abs(x[6:])).max()), collisions=collisions,
            unchanged_tail_node_error_mm=float(abs(p[10:]-self.oldpoints[10:]).max()),
            points_mm=p.tolist(), desired_points_mm=self.desired.tolist(), joint_angles_deg=np.degrees(x[6:]).tolist())
        return x.copy(), row


def main():
    resume = BASE.exists()
    BASE.mkdir(exist_ok=True)
    with (PARENT/'trial_ledger.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; history = list(reader)
    if len(history) != 15:
        raise RuntimeError('Expected 15 preserved dynamics; reconcile before registering')
    if not resume:
        with (BASE/'trial_ledger.csv').open('w', encoding='utf8', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        shutil.copy2(PARENT/'trial_ledger.csv', BASE/'historical_15_trial_ledger.csv')
        shutil.copy2(OLD/'initial_boundary_pose.npz', BASE/'initial_boundary_pose.npz')
        shutil.copy2(ORIGINAL, BASE/'original_reference.npz')
        write(BASE/'revision_registry.json', dict(status='OFFLINE', prior_dynamics=15, new_dynamics=0,
            new_budget=4, total_budget=19, total_dynamics=15, further_dynamics_authorized=True,
            scope='Target-only open-insert-close action; not full handoff prerequisite', full_V3_path='NOT_READY'))
    else:
        if (BASE/'reference_discrete.npz').exists():raise FileExistsError('Completed geometry exists')
        write(BASE/'fixed_tail_constraint_repair.json',dict(physics_steps=0,
            issue='Fixed downstream source nominal conservative gap .0485037mm cannot reach .05mm by front changes',
            repair='Constrain modifiable front modules0..9; fixed tail retains exact original geometry; all finite actual collisions still checked',
            new_radius_or_initial_search=False, preserved='geometry_fixed_tail_constraint_attempt0.json + reference_fixed_tail_constraint_attempt0.npz'))
    write(BASE/'geometry_plan.json', dict(physics_steps=0, radius_mm=[95,130,95], opening_mm=35,
        pipe_radius_mm=56.5, maximum_radial_shell_extent_mm=29.14032,
        amplitude_basis='35mm > body radial half-width 29.14mm, visible 36.8% change; R130 shell clearance about44.4mm before finite-chord/fit corrections',
        taper_material_length_mm=129., target_initial_material_length_mm=451.5,
        target_final_material_length_mm=532.125, active_geometry='J1..J10; downstream link10/tail world geometry stays equal old offline shape; actual root never commanded',
        source_reference='unchanged original J11..J20; source R95; no release',
        planned_seconds=dict(entry=5,opening=12,inserting=45,closing=12,post_closing_hold=10,resume_closed=45,end_holding=10),
        feed_range_mm=[0,161.25], opening_insert_endpoint_mm=80.625,
        two_questions='Open-close at former stable middle point demonstrates action only; resume to original failed endpoint tests later stability separately',
        radius_variable='unit-arclength target helix, fixed pitch per radian; R(sigma)=95+a*smooth(sigma/129)',
        offline_desired='saved original finite reference points plus new-minus-old ideal target curve; initial exact old reference',
        limitation='geometric hypothesis; shape/cloud prior is not a measurement of pipe axis',
        self_fit_diagnostics='free cylinder radius/axis separate from distance to known pipe axis, reject ill-conditioned fits',
        no_parameter_scan=True))
    fit = OpeningFit()
    grid = np.unique(np.r_[np.linspace(0,35,9), 35+fit.old.feeds[fit.old.feeds<=80.625],
                           115.625+np.linspace(0,35,9),
                           TargetOpenCloseReference.close_end+fit.old.feeds[fit.old.feeds>80.625]-80.625])
    states=[]; rows=[]; seed=fit.old.state(0.)
    if resume:
        with np.load(BASE/'reference_partial.npz') as z:
            states=list(z['states'].copy()); oldgrid=z['action_mm'].copy()
        rows=json.loads((BASE/'geometry_progress.json').read_text(encoding='utf8'))['rows']
        if not np.array_equal(grid[:len(states)],oldgrid):raise RuntimeError('Repair changed saved prefix')
        seed=states[-1]
    for action in grid[len(states):]:
        seed, row = fit.solve(float(action), seed)
        states.append(seed); rows.append(row)
        write(BASE/'geometry_progress.json',dict(physics_steps=0, rows=rows))
        np.savez_compressed(BASE/'reference_partial.npz',action_mm=grid[:len(states)], states=states)
        print(json.dumps({k:row.get(k) for k in ['action_mm','feed_mm','opening_mm','desired_front_rms_mm','minimum_conservative_gap_mm','unchanged_tail_node_error_mm'] }), flush=True)
        if row.get('collisions') or row.get('desired_front_rms_mm',0)>15:
            write(BASE/'geometry_result.json',dict(status='NOT_READY', physics_steps=0, failure=row,
                reason='Finite front geometry not admitted; no dynamics'))
            return
    np.savez_compressed(BASE/'reference_discrete.npz',action_mm=grid,states=states)
    reference=TargetOpenCloseReference(BASE/'reference_discrete.npz', ORIGINAL)
    failures=[]; maxangle=0.; lengtherror=0.; deviations=[]
    for a in np.unique(np.r_[np.arange(0,reference.action_end,.25),grid,reference.action_end]):
        x=reference.state(float(a));fit.fit.set(x);mujoco.mj_forward(fit.fit.m,fit.fit.d)
        contacts=[dict(geom1=fit.fit.m.geom(c.geom1).name, geom2=fit.fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.fit.d.contact if c.dist<0]
        if contacts:failures.append(dict(action_mm=float(a),contacts=contacts))
        maxangle=max(maxangle,float(np.degrees(abs(x[6:])).max()))
        p=points(fit.fit.m,fit.fit.d)
        lengtherror=max(lengtherror,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
    ready=not failures and maxangle<=89.001 and lengtherror<1e-6
    report=dict(status='READY_FOR_TIMING' if ready else 'NOT_READY', physics_steps=0,
        interpolation_failures=failures,maximum_joint_deg=maxangle,maximum_length_error_mm=lengtherror,
        key_poses=rows, finite_sampling='0.25 action mm plus knots; not continuum proof',full_V3_path='NOT_READY')
    write(BASE/'geometry_result.json', report)
    if ready:shutil.copy2(BASE/'reference_discrete.npz',BASE/'reference_ready.npz')
    print(json.dumps(dict(status=report['status'], failures=len(failures),physics_steps=0)),flush=True)


if __name__=='__main__':
    main()
