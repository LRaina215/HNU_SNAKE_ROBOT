"""Check whether the growing acquired coil supplies an independent axis fit.

Fit encoder-frame points using the previous estimated prefix, never the true
partition. Simulator pose/axes score the completed fit only.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import pipe_frame,write_json
from snake_control.encoder_pipe_geometry import cylinder


def run(folder):
    _,z,m,_,meta=context(folder)
    c=json.loads((folder/'controls.json').read_text(encoding='utf8'))
    qpos=z['qpos'];time=z['time']-z['time'][0]
    qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
    own=mujoco.MjData(m);own.qpos[:7]=[0,0,0,1,0,0,0]
    corner,_,target,_=pipe_frame(m);rows=[]
    for ci in range(5,len(c),5):
        info=c[ci-1];at=c[ci]['time']-.2
        if at>110:break
        prefix=info['actual_prefix']
        if prefix<8:continue
        i=int(np.argmin(abs(time-at)));q=qpos[i]
        own.qpos[qadr]=q[qadr];mujoco.mj_kinematics(m,own)
        points=1000*np.vstack([own.site('s_head').xpos,*[own.joint('joint'+str(j)).xanchor for j in range(1,21)],own.site('s_tail').xpos])
        fit=cylinder(points[:prefix+2],meta['settings']['radius'])
        est=info['geometry_observer']['estimate'];axis=fit['axis']
        if axis@est['target']<0:axis=-axis
        rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q[3:7]);rot=rot.reshape(3,3)
        angle=lambda a,b:float(np.degrees(np.arccos(np.clip(a@b,-1,1))))
        rows.append(dict(time=float(time[i]),prefix=prefix,radial_rms_mm=fit['radial_rms_mm'],
            radius_mm=fit['radius_mm'],target_error_deg=angle(rot@axis,target),
            prior_error_deg=angle(rot@est['target'],target),
            target_offset_mm=float(np.linalg.norm(np.cross(rot@fit['axis_point_mm']+1000*q[:3]-corner,target)))))
    for threshold in (8,10,12,15):
        sub=[r for r in rows if r['prefix']>=threshold]
        print(folder.name,'prefix>=',threshold,'frames',len(sub),'median/max axis error',np.median([r['target_error_deg'] for r in sub]),max(r['target_error_deg'] for r in sub),flush=True)
    write_json(folder.parent/(folder.name+'_coil_refit_probe.json'),dict(rows=rows,
        limitation='Stateless fit on recorded motions, inferred preceding prefix; small residual alone is not an observability certificate.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    run(p.parse_args().folder)
