"""Stage-2 read-only axis observability probe on the two captured shapes.

Estimation sees encoder-derived robot points in an arbitrary head frame only.
Known pipe axes enter the scoring wrapper, never the estimator. These two
positive captures cannot establish contact detection or classifier accuracy.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from segmented_pole_experiment import ROOT, HORIZONTAL, VERTICAL, pipe_frame, write_json
from snake_control.mujoco_checkpoint import load_checkpoint
from snake_control.curved_axis_compliance import window_axis

OUT = ROOT/'experiments/segmented_pole_20261005/feedback_takeover'


def fit(points):
    links = np.diff(points, axis=0)
    tangents = links/np.linalg.norm(links, axis=1)[:, None]
    axis, flatness = window_axis(tangents)
    h = np.eye(3)[np.argmin(abs(axis))]
    h -= h@axis*axis; h /= np.linalg.norm(h)
    plane = np.column_stack((h, np.cross(axis, h)))
    xy = points@plane
    solution = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))),
                               np.sum(xy*xy, axis=1), rcond=None)[0]
    radial = xy-solution[:2]
    distances = np.linalg.norm(radial, axis=1)
    angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    return dict(axis=axis, flatness=float(flatness), radius_mm=float(distances.mean()),
        circle_rms_mm=float(np.sqrt(np.mean((distances-distances.mean())**2))),
        signed_wrap_turns=float((angles[-1]-angles[0])/(2*np.pi)))


def fit_cylinder(points, nominal_radius):
    """Fit module centres without assuming constant pitch; no truth inputs.

    The three tangent-covariance eigenvectors are starting guesses, not known
    pipe directions. Bounds come from the commanded nominal coil radius.
    A small radial residual alone cannot certify axis identifiability.
    """
    centers=(points[:-1]+points[1:])/2
    centered=centers-centers.mean(axis=0)
    links=np.diff(points,axis=0)
    tangents=links/np.linalg.norm(links,axis=1)[:,None]
    _,vectors=np.linalg.eigh(np.cov(tangents.T,bias=True))
    candidates=[]
    for seed in vectors.T:
        h=np.eye(3)[np.argmin(abs(seed))];h-=h@seed*seed;h/=np.linalg.norm(h)
        v=np.cross(seed,h)
        def decode(x):
            axis=seed+x[0]*h+x[1]*v;axis/=np.linalg.norm(axis)
            e=h-(h@axis)*axis;e/=np.linalg.norm(e)
            center=x[2]*e+x[3]*np.cross(axis,e)
            return axis,center
        def residual(x):
            axis,center=decode(x)
            return np.linalg.norm(np.cross(centered-center,axis),axis=1)-x[4]
        fit=least_squares(residual,[0.,0.,0.,0.,nominal_radius],
            bounds=([-5,-5,-3*nominal_radius,-3*nominal_radius,.6*nominal_radius],
                    [5,5,3*nominal_radius,3*nominal_radius,1.8*nominal_radius]),
            x_scale=[.4,.4,nominal_radius,nominal_radius,nominal_radius],
            loss='soft_l1',f_scale=5.,max_nfev=300)
        axis,center=decode(fit.x)
        candidates.append(dict(axis=axis,axis_point_mm=center+centers.mean(axis=0),
            radius_mm=float(fit.x[4]),radial_rms_mm=float(np.sqrt(np.mean(residual(fit.x)**2))),
            objective=float(fit.cost),solver_success=bool(fit.success)))
    return min(candidates,key=lambda row:row['objective'])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [];cylinder_rows=[]
    for name, checkpoint in [('horizontal', HORIZONTAL), ('vertical_down', VERTICAL)]:
        m, captured, meta, _ = load_checkpoint(checkpoint, verify_hashes=False)
        qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        encoder = captured.qpos[qadr].copy()
        robot = mujoco.MjData(m)
        robot.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
        robot.qpos[qadr] = encoder
        mujoco.mj_kinematics(m, robot)
        points = 1000*np.vstack([robot.site('s_head').xpos,
            *[robot.joint('joint'+str(i)).xanchor for i in range(1, 21)], robot.site('s_tail').xpos])
        # From here the known environment is used only to score angular error.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, captured.qpos[3:7])
        rotation = rotation.reshape(3, 3)
        world = 1000*np.vstack([captured.site('s_head').xpos,
            *[captured.joint('joint'+str(i)).xanchor for i in range(1,21)],captured.site('s_tail').xpos])
        reconstructed=points@rotation.T+1000*captured.qpos[:3]
        frame_error=float(np.max(np.linalg.norm(reconstructed-world,axis=1)))
        print('FRAME_CHECK',name,frame_error,'mm')
        assert frame_error<1e-6
        _, source, target, _ = pipe_frame(m)
        for side, truth in [('front', target), ('rear', source)]:
            for joints in range(3, 12):
                block = points[:joints+2] if side == 'front' else points[-joints-2:]
                result = fit(block)
                error = np.degrees(np.arccos(np.clip(abs((rotation@result['axis'])@truth), 0., 1.)))
                rows.append(dict(scene=name, side=side, window_joints=joints,
                    axis_error_deg=float(error), **result))
                if joints>=5:
                    cylinder=fit_cylinder(block,meta['settings']['radius'])
                    error=np.degrees(np.arccos(np.clip(abs((rotation@cylinder['axis'])@truth),0.,1.)))
                    cylinder_rows.append(dict(scene=name,side=side,window_joints=joints,
                        axis_error_deg=float(error),**cylinder))
    write_json(OUT/'initial_geometry_probe.json', dict(rows=rows,
        estimator_inputs='robot link geometry and joint encoders only; arbitrary identity head pose',
        scoring_only='known pipe axes and captured head orientation',
        limitations=['Shape fitting does not establish contact or support.',
            'Two successful starting captures only; no false-positive evaluation yet.',
            'Candidate windows are reported, not chosen using their truth error.']))
    write_json(OUT/'initial_cylinder_probe.json',dict(rows=cylinder_rows,
        estimator_inputs='Encoder-derived module centres, commanded nominal radius and robot geometry only.',
        selection='Three covariance seeds; lowest robust radial objective. No pipe-axis truth used in selection.',
        limitations=['This does not establish contact, support, uniqueness or online stability.',
            'Only two positive captures; numerical fit may be confident and wrong.']))
    for row in rows:
        print(row['scene'], row['side'], row['window_joints'],
              'flatness=%.3f radius=%.1f circle_rms=%.1f wrap=%.2f axis_error=%.1f' %
              (row['flatness'], row['radius_mm'], row['circle_rms_mm'],
               row['signed_wrap_turns'], row['axis_error_deg']))
    for row in cylinder_rows:
        print('CYLINDER',row['scene'],row['side'],row['window_joints'],
              'radius=%.1f rms=%.1f axis_error=%.1f' %
              (row['radius_mm'],row['radial_rms_mm'],row['axis_error_deg']))


if __name__ == '__main__':
    run()
