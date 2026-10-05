"""Reproducible 20-joint pitch feasibility and geometry gates. Never drives hardware.

Run with MuJoCo 3.3.0. Outputs are separate from the old 26-joint baseline.
This is an offline gate, not a climbing success experiment.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess

import mujoco
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'codes')]
from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
from snake_control.curved_axis_compliance import joint_segments
from snake_control.local_pitch_geometry import VariablePitchBackbone, estimate_local_radius, pitch_rule
from snake_control.pole_compliance import ComplianceParameters, build_backbone

LINK, JOINTS = 64.5, 20
ALPHA = math.atan2(220., 2*np.pi*81.5)
SCENES = ('pipe_straight_small', 'pipe_straight_large', 'pipe_taper_horizontal', 'real_s2l')


def path_for(scene):
    return ROOT/'mujoco_model'/('Snake_Robot_o20_Real_Pipe_S2L.xml' if scene == 'real_s2l'
                                else f'Snake_Robot_o20_Pipe_{scene[5:]}.xml')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_rotation(source, target):
    x, y = source-source.mean(0), target-target.mean(0)
    u, _, v = np.linalg.svd(x.T@y)
    rotation = u @ np.diag([1., 1., np.linalg.det(u@v)]) @ v
    return (source-source.mean(0))@rotation+target.mean(0)


def robot_points(model, data):
    return np.vstack([data.site('s_head').xpos,
                      [data.joint(f'joint{i}').xanchor for i in range(1, JOINTS+1)],
                      data.site('s_tail').xpos])*1000


def geometry_from_points(points):
    """Synthetic straight-axis fixture; the axis is known only in this offline test."""
    xy = points[:, :2]
    vectors = np.diff(xy, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    turns = np.arctan2(np.cross(vectors[:-1], vectors[1:]), (vectors[:-1]*vectors[1:]).sum(1))
    winding = math.copysign(1., float(turns.sum()))
    source = [dict(joint_index=None, kind='line', length_mm=float(lengths[0]/2), radius_mm=0.,
                   turn_rad=0., start_mm=xy[0].tolist(), tangent=(vectors[0]/lengths[0]).tolist())]
    for j, turn in enumerate(turns, 1):
        for length, radius, angle in joint_segments(lengths[j-1]/2, lengths[j]/2, float(turn), winding, .01):
            source.append(dict(joint_index=j, kind='arc' if radius else 'line',
                               length_mm=float(length), radius_mm=float(radius), turn_rad=float(angle)))
    source.append(dict(joint_index=None, kind='line', length_mm=float(lengths[-1]/2), radius_mm=0., turn_rad=0.))
    return dict(points_mm=points.tolist(), curve=dict(valid=True, segments=source),
                config=dict(joint_count=JOINTS, head_length_mm=LINK, joints=[dict(joint_index=i+1,
                            axis='Yaw' if i%2==0 else 'Pitch', link_length_mm=LINK,
                            minimum_deg=-90., maximum_deg=90.) for i in range(JOINTS)]),
                projection_axis=dict(point_mm=[0.,0.,0.], horizontal=[1.,0.,0.],
                                     vertical=[0.,1.,0.], direction=[0.,0.,1.]))


def helix_points(radius, alpha=ALPHA):
    s = np.arange(JOINTS+2)*LINK
    u = s*np.cos(alpha)/radius
    return np.column_stack([radius*np.cos(u), radius*np.sin(u), s*np.sin(alpha)])


def scan(model, out):
    rows, traces = [], []
    data = mujoco.MjData(model)
    robot = [i for i in range(model.ngeom) if model.geom_bodyid[i] and
             not (model.geom(i).name or '').startswith('pipe')]
    pairs = [(x,y) for i,x in enumerate(robot) for y in robot[i+2:]]
    phases = np.linspace(0., 2*np.pi, 73)
    for radius in (85.5, 95., 110.5, 125., 140.):
        candidates = [(f'fixed_{deg:g}', math.radians(deg)) for deg in (12.,18.,math.degrees(ALPHA),28.)]
        candidates += [(f'rule_k{k:g}', float(pitch_rule(radius,k)[0])) for k in (0.,.5,1.)]
        for method, alpha in candidates:
            pitch = 2*np.pi*radius*math.tan(alpha)
            curve = HelixBackbone.from_radius_pitch(radius, pitch)
            target = helix_points(radius, alpha)
            q = np.array([curve.joint_angles(JOINTS, LINK, phase) for phase in phases])
            errors, turns, min_gap, worst_pair = [], [], .1, None
            for phase, angles in zip(phases, q):
                data.qpos[:]=0.; data.qpos[3]=1.; data.qpos[7:]=angles
                mujoco.mj_forward(model,data)
                points = robot_points(model,data)
                aligned = fit_rotation(points,target)
                errors.append(np.sqrt(np.mean(np.sum((aligned-target)**2,axis=1))))
                azimuth = np.unwrap(np.arctan2(aligned[:,1],aligned[:,0]))
                turns.append((azimuth[-1]-azimuth[0])/(2*np.pi))
                for x,y in pairs:
                    gap=float(mujoco.mj_geomDistance(model,data,x,y,.1,None))
                    if gap < min_gap:
                        min_gap,worst_pair=gap,[int(model.geom_bodyid[x]),int(model.geom_bodyid[y])]
            steps = np.array([curve.joint_angles(JOINTS,LINK,p-.15)-curve.joint_angles(JOINTS,LINK,p) for p in phases])
            row=dict(radius_mm=radius, candidate=method, alpha_deg=math.degrees(alpha), pitch_mm=pitch,
                     ideal_turns=(JOINTS+1)*LINK*math.cos(alpha)/(2*np.pi*radius),
                     ideal_turn_ceiling=(JOINTS+1)*LINK/(2*np.pi*radius),
                     discrete_turns_min=float(min(turns)),discrete_turns_max=float(max(turns)),
                     max_angle_deg=float(np.degrees(abs(q)).max()),
                     max_roll_step_deg=float(np.degrees(abs(steps)).max()),
                     fk_rms_mm_max=float(max(errors)),fk_rms_mm_median=float(np.median(errors)),
                     nonadjacent_gap_mm_lower_bound=min_gap*1000.,worst_pair=worst_pair,
                     feasible=bool(abs(q).max()<=np.pi/2 and min_gap>=-1e-8))
            rows.append(row)
            traces.append(q)
    with (out/'scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'scan_targets.npz',phases=phases,targets_rad=traces)
    return rows


def geometry_checks(out, model):
    geometry=geometry_from_points(helix_points(110.5))
    params=ComplianceParameters((.01,.02),ALPHA,False)
    worst=0.
    for alpha in (0., math.radians(12.), ALPHA, math.radians(28.)):
        old=build_backbone(geometry,ComplianceParameters(params.gain,alpha,False))
        new=VariablePitchBackbone(geometry,params,[alpha,alpha])
        for phase in np.linspace(0,2*np.pi,17):
            worst=max(worst,float(abs(new.joint_angles_numerical(phase)-old.joint_angles(phase)).max()))
        # Independent old sampler compared at original segment sampling positions.
        positions=np.r_[0.,np.concatenate([np.linspace(left,right,max(1,math.ceil((right-left)/2))+1)[1:]
                            for left,right in zip(old.bounds[:-1],old.bounds[1:])])]
        assert np.max(abs(new.sample_at(positions,False)-old.sample(2.,False))) < 1e-6
    for winding_sign in (1.,-1.):
        for axis_sign in (1.,-1.):
            mirrored=copy.deepcopy(geometry)
            for segment in mirrored['curve']['segments']:segment['turn_rad']*=winding_sign
            for point in mirrored['points_mm']:point[2]*=axis_sign
            for alpha in (-ALPHA,ALPHA):
                old=build_backbone(mirrored,ComplianceParameters(params.gain,alpha,False))
                new=VariablePitchBackbone(mirrored,params,[alpha,alpha])
                for phase in (0.,.7,2.1):
                    worst=max(worst,float(abs(new.joint_angles_numerical(phase)-old.joint_angles(phase)).max()))
    s_nodes=np.linspace(0,(JOINTS+1)*LINK,9)
    radii=85.5+25.*(.5+.5*np.tanh((s_nodes-s_nodes[-1]/2)/150.))
    alpha,_=pitch_rule(radii,1.)
    new=VariablePitchBackbone(geometry,params,alpha,s_nodes)
    s=np.linspace(0,new.bounds[-1],math.ceil(new.bounds[-1]/.1)+1)
    path=new.sample_at(s,False)
    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    length_error=abs(length-new.bounds[-1])/new.bounds[-1]
    # Independently integrate T and a Bishop frame directly in 3D.
    rate=new.turn_rates[0];a=new.axial_sign;w=new.winding
    aa=float(new.alpha(0));heading=new.headings[0]
    t=np.array([math.cos(heading),math.sin(heading),0.])
    normal=np.array([-math.sin(heading),math.cos(heading),0.])*w
    tangent=math.cos(aa)*t+np.array([0.,0.,a*math.sin(aa)])
    e2=np.cross(tangent,normal)
    state=np.r_[new.flat.start_xy,new.flat.start_z,normal,e2,0.,0.]
    dense=[]
    for index,rate in enumerate(new.turn_rates):
        left,right=new.bounds[index:index+2];heading=new.headings[index]
        def rhs(x,y,rate=rate,left=left,heading=heading):
            alpha=float(new.alpha(x));ap=float(new.alpha_derivative(x));theta=heading+rate*(x-left)
            planar_t=np.array([math.cos(theta),math.sin(theta),0.])
            planar_n=np.array([-math.sin(theta),math.cos(theta),0.])
            T=math.cos(alpha)*planar_t+np.array([0.,0.,a*math.sin(alpha)])
            Tp=-math.sin(alpha)*ap*planar_t+math.cos(alpha)*rate*planar_n+np.array([0.,0.,a*math.cos(alpha)*ap])
            c1,c2=np.dot(Tp,y[3:6]),np.dot(Tp,y[6:9])
            return np.r_[T,-c1*T,-c2*T,c1,c2]
        sol=solve_ivp(rhs,(left,right),state,rtol=1e-10,atol=1e-11,max_step=1.,dense_output=True)
        assert sol.success
        state=sol.y[:,-1];dense.append(sol.sol)
    boundaries=np.arange(JOINTS+2)*LINK
    independent=[]
    for x in boundaries:
        idx=min(len(dense)-1,np.searchsorted(new.bounds,x,side='right')-1)
        independent.append(dense[idx](x))
    independent=np.array(independent)
    bishop_point_error=float(np.max(np.linalg.norm(independent[:,:3]-new.sample_at(boundaries,False),axis=1)))
    difference=independent[2:,9:]-independent[:-2,9:]
    independent_q=-np.where(np.arange(JOINTS)%2==0,difference[:,1],difference[:,0])
    bishop_angle_error=float(abs(independent_q-new.joint_angles(0)).max())
    # Quantify the mistake of omitting alpha': same connection, zero b2.
    from scipy.integrate import quad
    naive=[]
    for j in range(1,JOINTS+1):
        lo,hi=boundaries[j-1],boundaries[j+1]
        def integrand(x):
            idx=min(len(new.turn_rates)-1,np.searchsorted(new.bounds,x,side='right')-1)
            phase=new.evaluate([x])[0,3]
            return -w*math.cos(float(new.alpha(x)))*new.turn_rates[idx]*(math.sin(phase) if j%2 else math.cos(phase))
        naive.append(quad(integrand,lo,hi,points=new.bounds[(new.bounds>lo)&(new.bounds<hi)],epsabs=1e-9)[0])
    naive_error=float(abs(np.array(naive)-new.joint_angles(0)).max())
    data=mujoco.MjData(model);old=build_backbone(geometry,params)
    rows=[]
    for phase in np.linspace(0,2*np.pi,73):
        row=dict(phase=float(phase))
        for label,curve in (('fixed',old),('variable',new)):
            q=curve.joint_angles(float(phase))
            data.qpos[:]=0;data.qpos[3]=1;data.qpos[7:]=q;mujoco.mj_forward(model,data)
            target=new.sample_at(boundaries,False) if label=='variable' else VariablePitchBackbone(geometry,params,[ALPHA,ALPHA]).sample_at(boundaries,False)
            points=robot_points(model,data)
            error=np.linalg.norm(fit_rotation(points,target)-target,axis=1)
            row[label+'_fk_rms_mm']=float(np.sqrt(np.mean(error**2)))
            row[label+'_max_angle_deg']=float(np.degrees(abs(q)).max())
        rows.append(row)
    np.savez_compressed(out/'variable_geometry.npz',s_mm=s,target_mm=path,alpha_nodes=alpha,node_s=s_nodes,
                        alpha_rad=new.alpha(s),kappa=new.curvature_torsion(s)[0],tau=new.curvature_torsion(s)[1])
    (out/'fk_variable.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    checks=dict(constant_max_error_rad=worst,length_relative_error=length_error,
                independent_bishop_point_error_mm=bishop_point_error,
                independent_bishop_angle_error_rad=bishop_angle_error,
                missing_alpha_derivative_error_rad=naive_error,
                fixed_fk_rms_mm_max=max(r['fixed_fk_rms_mm'] for r in rows),
                variable_fk_rms_mm_max=max(r['variable_fk_rms_mm'] for r in rows),
                variable_max_angle_deg=max(r['variable_max_angle_deg'] for r in rows))
    checks['continuous_gate_passed']=worst<=1e-6 and length_error<=.001 and bishop_point_error<=.001 and bishop_angle_error<=1e-6
    return checks


def estimator_checks(out,model):
    axis=dict(point_mm=[0.,0.,0.],horizontal=[1.,0.,0.],vertical=[0.,1.,0.],direction=[0.,0.,1.])
    rows=[]
    data=mujoco.MjData(model)
    for window in (7,9,13):
        for radius in (85.5,110.5):
            for seed in (1,2,3):
                # Explicit synthetic coordinate noise, not an encoder model.
                points=helix_points(radius)+np.random.default_rng(seed).normal(0,.1,(JOINTS+2,3))
                estimates=estimate_local_radius(points,axis,window)
                error=[r['radius_mm']-radius for r in estimates if r['valid']]
                rows.append(dict(case='ideal_noise_0.1mm',window=window,radius_mm=radius,seed=seed,
                                 valid_fraction=len(error)/len(estimates),rmse_mm=float(np.sqrt(np.mean(np.array(error)**2))) if error else None))
        # Deliberately under-covered arc must fail, not fabricate a pipe radius.
        points=helix_points(10000.)
        estimates=estimate_local_radius(points,axis,window)
        rows.append(dict(case='undercovered',window=window,valid_fraction=sum(x['valid'] for x in estimates)/len(estimates)))
        s=np.arange(JOINTS+2)*LINK;radius=85.5+25*(.5+.5*np.tanh((s-s[-1]/2)/90.))
        u=np.zeros_like(s);u[1:]=np.cumsum(LINK*np.cos(ALPHA)/(radius[:-1]+radius[1:])*2)
        points=np.column_stack([radius*np.cos(u),radius*np.sin(u),s*np.sin(ALPHA)])
        estimates=estimate_local_radius(points,axis,window)
        errors=[r['radius_mm']-radius[i] for i,r in enumerate(estimates) if r['valid']]
        rows.append(dict(case='mixed_radius_transition',window=window,valid_fraction=len(errors)/len(estimates),
                         rmse_mm=float(np.sqrt(np.mean(np.array(errors)**2))) if errors else None,
                         max_error_mm=float(max(abs(np.array(errors)))) if errors else None,estimates=estimates))
        # Protocol conversion is 4096 counts/revolution (Kinematics.cs), so the
        # quantization diagnostic is +/- half a count. This is not a calibrated
        # encoder accuracy claim. Isolate quantization from Yamada's own error.
        for radius in (85.5,110.5):
            reference=helix_points(radius)
            for phase in (0.,.7,2.1,4.5):
                exact=HelixBackbone.from_radius_pitch(radius,2*np.pi*radius*np.tan(ALPHA)).joint_angles(JOINTS,LINK,phase)
                data.qpos[:]=0.;data.qpos[3]=1.;data.qpos[7:]=exact;mujoco.mj_forward(model,data)
                noiseless=fit_rotation(robot_points(model,data),reference)
                baseline=estimate_local_radius(noiseless,axis,window)
                for seed in (1,2,3):
                    noise=np.random.default_rng(seed).uniform(-np.pi/4096,np.pi/4096,JOINTS)
                    data.qpos[7:]=exact+noise;mujoco.mj_forward(model,data)
                    noisy=fit_rotation(robot_points(model,data),reference)
                    estimates=estimate_local_radius(noisy,axis,window)
                    delta=[r['radius_mm']-b['radius_mm'] for r,b in zip(estimates,baseline) if r['valid'] and b['valid']]
                    bias=[r['radius_mm']-radius for r in estimates if r['valid']]
                    rows.append(dict(case='fk_half_count_noise',window=window,radius_mm=radius,phase=phase,seed=seed,
                                     valid_fraction=sum(r['valid'] for r in estimates)/len(estimates),
                                     quantization_rmse_mm=float(np.sqrt(np.mean(np.array(delta)**2))) if delta else None,
                                     design_radius_rmse_mm=float(np.sqrt(np.mean(np.array(bias)**2))) if bias else None))
    (out/'estimator.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/pitch_adaptation_20261003/stage0')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if mujoco.__version__!='3.3.0':
        raise RuntimeError('Use the frozen MuJoCo 3.3.0 environment')
    models={scene:mujoco.MjModel.from_xml_path(str(path_for(scene))) for scene in SCENES}
    for scene,model in models.items():
        assert model.nu==20 and model.opt.timestep==.001,(scene,model.nu,model.opt.timestep)
        assert np.allclose(model.actuator_gainprm[:,0],127.)
        assert np.allclose(model.actuator_forcerange[:,1],2.355)
    provenance=dict(stage='offline_geometry_only',joints=20,mujoco=mujoco.__version__,numpy=np.__version__,
                    python=sys.version,python_executable=sys.executable,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    models={s:dict(path=str(path_for(s)),sha256=sha(path_for(s))) for s in SCENES},
                    exe_sha256=sha(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe'),
                    sources={p:sha(ROOT/p) for p in ('snake_control/local_pitch_geometry.py','validation/pitch_adaptation_stage0.py','snake_control/pole_compliance.py')},
                    frozen_scan=dict(radii_mm=[85.5,95.,110.5,125.,140.],phase_samples=73,gain=[.01,.02],
                                     encoder_noise='FK joint-angle perturbation +/- half of 4096-count revolution; protocol diagnostic, not calibrated encoder accuracy'))
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=scan(models['pipe_straight_small'],out)
    checks=geometry_checks(out,models['pipe_straight_small'])
    estimator=estimator_checks(out,models['pipe_straight_small'])
    encoder=[r for r in estimator if r['case']=='fk_half_count_noise']
    quantization=[r['quantization_rmse_mm'] for r in encoder if r['quantization_rmse_mm'] is not None]
    report=dict(provenance=provenance,scan_cases=len(rows),feasible_cases=sum(r['feasible'] for r in rows),
                scan_fk_rms_mm_max=max(r['fk_rms_mm_max'] for r in rows),geometry=checks,
                estimator_ideal_rmse_mm_max=max(r['rmse_mm'] for r in estimator if r['case']=='ideal_noise_0.1mm'),
                estimator_undercovered_rejected=all(r['valid_fraction']==0 for r in estimator if r['case']=='undercovered'),
                encoder_half_count_valid_fraction_min=min(r['valid_fraction'] for r in encoder),
                encoder_half_count_valid_fraction_max=max(r['valid_fraction'] for r in encoder),
                encoder_half_count_radius_delta_rmse_mm_max=max(quantization) if quantization else None,
                stage1_ready=False,reason='Continuous geometry passed; the circle estimator rejects entire FK windows. Real feedback pilots are required before freezing formal trials.')
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
