"""Test whether a short active window stabilises target-line estimation.

All probe configurations are aligned on the held rear robot points using
encoder FK. This is an approximate gauge, not measured world odometry. True
pipe pose is only used after fitting to score errors and expose that limitation.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
import numpy as np
import mujoco
from scipy.optimize import least_squares
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context
from snake_control.encoder_pipe_proximity import EncoderPipeProximity


def alignment(x,y):
    u,_,vt=np.linalg.svd((x-x.mean(axis=0)).T@(y-y.mean(axis=0)))
    fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T)
    rot=vt.T@fix@u.T
    return rot,y.mean(axis=0)-rot@x.mean(axis=0)


def fit(query,samples,initial,front,temporal,absolute_difference=False):
    c=np.asarray(initial['corner_mm']);a=np.asarray(initial['target']);source=np.asarray(initial['source']);r=initial['pipe_radius_mm']
    h=np.eye(3)[np.argmin(abs(a))];h-=h@a*a;h/=np.linalg.norm(h);v=np.cross(a,h)
    q,p=samples[0]
    gap=query.gaps(q,p,c,source,a,r)[1]
    selected=(gap<5.)&(np.arange(21)<=front)
    active=samples if temporal else samples[:1]
    def decode(x):
        direction=a+x[2]*h+x[3]*v;direction/=np.linalg.norm(direction)
        return c+x[0]*h+x[1]*v,direction
    def residual(x):
        corner,direction=decode(x);rows=[]
        for q,p in active:
            g=query.gaps(q,p,corner,source,direction,r)[1]
            rows.extend([g[selected]+.5,2*np.minimum(g+2.,0.)])
        return np.r_[np.concatenate(rows)/np.sqrt(len(active)),x[:2]/20.,x[2:]/.1]
    def jacobian(x):
        steps=np.array([.05,.05,.0005,.0005]);columns=[]
        for i,step in enumerate(steps):
            delta=np.zeros(4);delta[i]=step
            columns.append((residual(x+delta)-residual(x-delta))/(2*step))
        return np.column_stack(columns)
    solved=least_squares(residual,np.zeros(4),jac=jacobian if absolute_difference else '2-point',bounds=([-30,-30,-.3,-.3],[30,30,.3,.3]),
        x_scale=[5,5,.05,.05],diff_step=1e-3,loss='soft_l1',f_scale=1.,max_nfev=60)
    corner,direction=decode(solved.x)
    return dict(corner_mm=corner,target=direction,selected_modules=np.flatnonzero(selected),
        cost=float(solved.cost),success=bool(solved.success),parameter_change=solved.x,nfev=solved.nfev)


def run(absolute_difference):
    root=ROOT/'experiments/segmented_pole_20261005/active_takeover_multisine'
    trials=json.loads((root/'probe_trials.json').read_text(encoding='utf8'))['trials'];rows=[]
    for scene in ('horizontal','vertical'):
        name='horizontal_encoder_live_whole_coil' if scene=='horizontal' else 'vertical_encoder_live_whole_coil'
        folder=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'/name
        _,z,m,_,_=context(folder);times=z['time']-z['time'][0];qpos=z['qpos']
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        own=mujoco.MjData(m);own.qpos[:7]=[0,0,0,1,0,0,0];qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
        query=EncoderPipeProximity(m);true_corner,_,true_axis,_=pipe_frame(m)
        def points(q):
            own.qpos[qadr]=q;mujoco.mj_kinematics(m,own)
            return 1000*np.vstack([own.site('s_head').xpos,*[own.joint('joint'+str(j)).xanchor for j in range(1,21)],own.site('s_tail').xpos])
        for trial in trials:
            if trial['scene']!=scene or trial['pattern']!='multisine' or trial['target_removed']:continue
            at=trial['snapshot_seconds'];info=min(controls,key=lambda c:abs(c['time']-.2-at));initial=info['geometry_observer']['estimate']
            records=trial['records'];anchor=points(records[0]['q']);samples=[];gauge=[]
            for record in records[::10]:
                p=points(record['q']);rot,shift=alignment(p[-8:],anchor[-8:]);aligned=p@rot.T+shift
                gauge.append(float(np.sqrt(np.mean((aligned[-8:]-anchor[-8:])**2))))
                samples.append((np.asarray(record['q']),aligned))
            fits=[dict(mode='initial',corner_mm=np.asarray(initial['corner_mm']),target=np.asarray(initial['target']))]
            for temporal in (False,True):fits.append(dict(mode='temporal' if temporal else 'static',**fit(query,samples,initial,trial['front'],temporal,absolute_difference)))
            # Initial physical pose appears below only in the scorer.
            i=int(np.argmin(abs(times-at)));root_rot=np.zeros(9);mujoco.mju_quat2Mat(root_rot,qpos[i,3:7]);root_rot=root_rot.reshape(3,3)
            for row in fits:
                a=root_rot@row['target'];c=root_rot@row['corner_mm']+1000*qpos[i,:3]
                row.update(target_axis_error_deg=float(np.degrees(np.arccos(np.clip(a@true_axis,-1,1)))),
                    target_line_offset_mm=float(np.linalg.norm(np.cross(c-true_corner,true_axis))))
            row=dict(scene=scene,time=at,front=trial['front'],held_rear_registration_rms_mm=gauge,fits=fits);rows.append(row)
            print(scene,at,[(r['mode'],round(r['target_axis_error_deg'],3),round(r['target_line_offset_mm'],3)) for r in fits],flush=True)
            write_json(root/('geometry_window_absolute_difference.json' if absolute_difference else 'geometry_window.json'),dict(rows=rows,
                limitation='Diagnostic fits of existing probe records. Held-rear registration is approximate and does not prove the rear is anchored. No closed-loop crossing yet.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--absolute-difference',action='store_true')
    run(p.parse_args().absolute_difference)
