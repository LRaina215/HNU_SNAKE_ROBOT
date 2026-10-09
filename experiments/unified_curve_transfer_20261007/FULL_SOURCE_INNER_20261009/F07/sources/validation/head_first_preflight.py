"""Offline geometry proposals only: never advances or resets physical simulation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco,json
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from snake_control.head_first_inner_route import HeadFirstInnerRoute
from snake_control.suspended_supply_transfer import WholeCurveMapper,nodes,LINK
from snake_control.suspended_transfer_route import R
from validation.full_source_inner_experiment import BASE,IMODEL
from validation.suspended_supply_experiment import write,default

def run():
    out=BASE/'HEAD_FIRST_LOWER_PREFLIGHT';out.mkdir(exist_ok=False)
    z=np.load(BASE/'F04/trace.npz');i=int(np.argmin(abs(z['sim_time']-14.75)))
    m=mujoco.MjModel.from_xml_path(str(IMODEL));d=mujoco.MjData(m);d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);p=nodes(d)
    # Hypothetical design seed, explicitly not an actual repositioning result.
    a=-np.pi/4-np.arctan2(p[0,2]-1,p[0,1]);p=(p-[0,0,1])@Rotation.from_rotvec([a,0,0]).as_matrix().T+[0,0,1];p[:,0]+=.95-p[0,0]
    route=HeadFirstInnerRoute(p,smooth_source=True);checks=[]
    def check(label,guide):
        mapper=WholeCurveMapper(m);pose,fit=mapper.initialize(guide);mapper.d.qpos[:]=pose;mujoco.mj_forward(m,mapper.d);actual=nodes(mapper.d)
        row=dict(label=label,rod_error_mm=float(abs(np.linalg.norm(np.diff(guide,axis=0),axis=1)-LINK).max()*1000),fit_rms_mm=float(np.sqrt(np.mean(np.sum((actual-guide)**2,axis=1)))*1000),maximum_joint_deg=float(np.degrees(abs(pose[mapper.qa]).max())),pipe_depth_mm=fit['pipe_depth_mm'],self_depth_mm=fit['self_depth_mm'],reference_points_mm=guide*1000,private_projected_points_mm=actual*1000)
        checks.append(row);print(json.dumps({k:v for k,v in row.items() if not k.endswith('points_mm')},default=default),flush=True)
    for pitch in [.18,.13,.08,.19,.30]:
        alpha=pitch/np.sqrt((2*np.pi*R)**2+pitch**2);beta=np.sqrt(1-alpha*alpha);h=brentq(lambda x:(alpha*x)**2+(2*R*np.sin(beta*x/(2*R)))**2-LINK**2,LINK,LINK*1.2);u=-np.arange(22)*h;phi=-np.pi/4+beta*u/R
        check('source_pitch_'+str(pitch),np.column_stack([.95+alpha*u,R*np.cos(phi),1+R*np.sin(phi)]))
    for s in [0,.025,.07,.15,.25,.45,.70,route.total]:check('head_shift_'+str(s),route.window(s)[0])
    summary=dict(physical_steps=0,seed='F04 junction transformed only in private design coordinates to a proposed lower inner exit; not actual motion or a physical reset',bridge_length_mm=route.bridge_length*1000,bridge_target_node_radius_min_mm=float(np.linalg.norm(route.connector[:,[0,2]]-[1.1,1],axis=1).min()*1000),maximum_rod_error_mm=max(x['rod_error_mm'] for x in checks),maximum_fit_rms_mm=max(x['fit_rms_mm'] for x in checks),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in checks),maximum_self_depth_mm=max(x['self_depth_mm'] for x in checks),maximum_joint_deg=max(x['maximum_joint_deg'] for x in checks),acceptance='exact rods <0.001mm, true20-hinge approximate fit <35mm, pipe <2mm, self <0.5mm, command hinges <=88deg',ready=all(x['rod_error_mm']<.001 and x['fit_rms_mm']<35 and x['pipe_depth_mm']<2 and x['self_depth_mm']<.5 and x['maximum_joint_deg']<=88.00001 for x in checks),interpretation='Approximate collision-aware joint fit; does not establish physical repositioning or world-coordinate tracking.')
    write(out/'checks.json',checks);write(out/'summary.json',summary);grid=np.linspace(route.knots[-1],route.total,3001);write(out/'proposed_curve.json',dict(stations_mm=grid*1000,points_mm=route.sample(grid)*1000));print(json.dumps(summary,default=default),flush=True)

if __name__=='__main__':run()
