"""Offline estimator repair audit; preserves all previous pitch trial sources.

True axes/radii below are scoring-only. No simulator, EXE or servo is started.
Reports alternative estimators including failures; does not release formal runs.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import hashlib
import json

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'codes'), str(ROOT/'validation')]
from snake_control.local_pitch_geometry import estimate_local_radius
from snake_control.local_pitch_feedback import estimate_module_radius
from pitch_adaptation_stage0 import (ALPHA, JOINTS, LINK, HelixBackbone,
                                     fit_rotation, helix_points, path_for, robot_points)


def summary(rows, expected):
    errors = np.array([row['radius_mm']-ref for row,ref in zip(rows,expected) if row['valid']])
    return dict(valid_fraction=float(np.mean([r['valid'] for r in rows])),
                fit_residual_median_mm=float(np.median([r['residual_mm'] for r in rows if r['residual_mm'] is not None])),
                accepted_error_rmse_mm=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                accepted_abs_error_p95_mm=float(np.percentile(abs(errors),95)) if len(errors) else None,
                accepted_abs_error_max_mm=float(max(abs(errors))) if len(errors) else None)


def truth_axis(scene):
    if scene=='pipe_taper_horizontal':
        return dict(point_mm=[0.,0.,1000.], horizontal=[0.,1.,0.],vertical=[0.,0.,1.],direction=[1.,0.,0.])
    return dict(point_mm=[0.,0.,0.], horizontal=[1.,0.,0.],vertical=[0.,1.,0.],direction=[0.,0.,1.])


def actual_window_radius(centres, axis, rows):
    # Reference is the average ACTUAL link-centre distance to the true straight
    # pipe axis over the identical window, not pipe radius + shell radius.
    p = (centres-np.asarray(axis['point_mm'])) @ np.column_stack([axis['horizontal'],axis['vertical']])
    radial = np.linalg.norm(p,axis=1)
    return np.array([np.mean(radial[r['lo']:r['hi']]) for r in rows])


def replay(out):
    base=ROOT/'experiments/pitch_adaptation_20261003'
    cases=[]
    for folder in ('matched_pilots_v2','estimator_repeats_v2'):
        cases += [p for p in (base/folder).iterdir() if '__D__' in p.name and (p/'result.json').exists()]
    results=[]
    for case in sorted(cases):
        result=json.loads((case/'result.json').read_text(encoding='utf8'))
        scene=result['scene'];true=truth_axis(scene)
        for window in (7,9,13):
            for sampler in ('joints','module_centres'):
                for projection in ('feedback_axis','true_axis_diagnostic'):
                    rows=[];expected=[];zero=[];times=[]
                    for file in sorted(case.glob('feedback_*.json.gz')):
                        with gzip.open(file,'rt',encoding='utf8') as f:g=json.load(f)
                        points=np.asarray(g['points_mm']);axis=g['projection_axis'] if projection=='feedback_axis' else true
                        samples=points if sampler=='joints' else (points[:-1]+points[1:])/2
                        fitted=estimate_local_radius(samples,axis,window)
                        rows+=fitted;expected.extend(actual_window_radius(samples,true,fitted))
                        zero.append(not any(r['valid'] for r in fitted));times.append(g['sim_time'])
                    stats=summary(rows,expected)
                    stats.update(scene=scene,seed=result['seed'],window=window,sampler=sampler,projection=projection,
                                 frames=len(times),zero_valid_frame_fraction=float(np.mean(zero)))
                    results.append(stats)
    (out/'feedback_ablation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    return results


def synthetic(out):
    axis=truth_axis('pipe_straight_small');rows=[]
    # Exact circle module-centre radius is R*cos(delta theta/2), since the
    # centre of a straight link lies on a chord, not on the design circle.
    for radius in (85.5,95.,110.5,125.,140.):
        for window in (7,9,13):
            p=helix_points(radius)
            target=radius*np.cos(LINK*np.cos(ALPHA)/(2*radius))
            estimates=estimate_module_radius(p,axis,window)
            errors=np.array([r['radius_mm']-target for r in estimates['modules']])
            assert np.max(abs(errors))<1e-6
            rows.append(dict(case='ideal_chord_centres',radius_mm=radius,window=window,
                             reference_radius_mm=float(target),max_error_mm=float(max(abs(errors)))))
    assert not any(r['valid'] for r in estimate_module_radius(helix_points(10000.),axis)['nodes'])
    # Continuous two-radius profile, deliberately crossing a window. References
    # are actual chord-centre radial distances. Pointwise error measures spatial
    # mixing separately from window-mean fit error.
    for width in (0.,90.,180.):
        s=np.arange(JOINTS+2)*LINK
        radii=85.5+25*((s>=s[-1]/2).astype(float) if width==0 else .5+.5*np.tanh((s-s[-1]/2)/width))
        angle=np.r_[0.,np.cumsum(2*LINK*np.cos(ALPHA)/(radii[:-1]+radii[1:]))]
        p=np.column_stack([radii*np.cos(angle),radii*np.sin(angle),s*np.sin(ALPHA)])
        centres=(p[:-1]+p[1:])/2;reference=np.linalg.norm(centres[:,:2],axis=1)
        for window in (7,9,13):
            fitted=estimate_module_radius(p,axis,window)['modules']
            row=summary(fitted,reference)
            row.update(case='two_radius_transition',width_mm=width,window=window,
                       window_mean_score=summary(fitted,actual_window_radius(centres,axis,fitted)),
                       estimates=fitted)
            rows.append(row)
    # Independent actual MuJoCo FK plus half-count angular perturbation.
    model=mujoco.MjModel.from_xml_path(str(path_for('pipe_straight_small')));data=mujoco.MjData(model)
    for radius in (85.5,110.5):
        for phase in np.linspace(0,2*np.pi,13):
            exact=HelixBackbone.from_radius_pitch(radius,2*np.pi*radius*np.tan(ALPHA)).joint_angles(JOINTS,LINK,float(phase))
            for seed in (1,2,3):
                noise=np.random.default_rng(seed).uniform(-np.pi/4096,np.pi/4096,JOINTS)
                data.qpos[:]=0.;data.qpos[3]=1.;data.qpos[7:]=exact+noise;mujoco.mj_forward(model,data)
                points=fit_rotation(robot_points(model,data),helix_points(radius));centres=(points[:-1]+points[1:])/2
                for window in (7,9,13):
                    fitted=estimate_module_radius(points,axis,window)['modules']
                    row=summary(fitted,actual_window_radius(centres,axis,fitted))
                    row.update(case='fk_half_count',design_radius_mm=radius,phase=float(phase),seed=seed,window=window)
                    rows.append(row)
    (out/'synthetic.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/pitch_adaptation_20261003/repair_v3/offline')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    replay_rows=replay(args.out);synthetic_rows=synthetic(args.out)
    result=dict(mujoco=mujoco.__version__,formal_released=False,
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),ROOT/'snake_control/local_pitch_feedback.py',ROOT/'snake_control/local_pitch_geometry.py')},
                small_pipe_module_centres=[r for r in replay_rows if r['scene']=='pipe_straight_small' and r['sampler']=='module_centres'],
                transition=[{k:v for k,v in r.items() if k!='estimates'} for r in synthetic_rows if r['case']=='two_radius_transition'])
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
