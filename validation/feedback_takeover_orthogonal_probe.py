"""Read-only right-angle branch fit: source estimated from encoders as well.

Right-angle, intersecting axes and comparable radii are explicit scene-family
priors. They are not known axis directions, contact labels or split indices.
This diagnostic reports candidates, not a validated takeover detector.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from scipy.optimize import least_squares
from feedback_takeover_initial_geometry import fit_cylinder, OUT
from segmented_pole_experiment import HORIZONTAL, VERTICAL, pipe_frame, write_json
from snake_control.mujoco_checkpoint import load_checkpoint


def fit_branch(centers, source, radius_prior=True):
    a=source['axis'];p=source['axis_point_mm'];r=source['radius_mm']
    h=np.eye(3)[np.argmin(abs(a))];h-=h@a*a;h/=np.linalg.norm(h)
    v=np.cross(a,h)
    z0=float(np.mean((centers-p)@a))
    def decode(x):
        d=np.cos(x[0])*h+np.sin(x[0])*v
        return d,p+x[1]*a
    def residual(x):
        d,c=decode(x)
        radial=np.linalg.norm(np.cross(centers-c,d),axis=1)-x[2]
        return np.r_[radial,(x[2]-r)*.5] if radius_prior else radial
    solutions=[]
    for phi in np.linspace(-np.pi,np.pi,12,endpoint=False):
        fitted=least_squares(residual,[phi,z0,r],
            bounds=([-10*np.pi,z0-3*r,.6*r],[10*np.pi,z0+3*r,1.8*r]),
            x_scale=[1.,r,r],loss='soft_l1',f_scale=5,max_nfev=200)
        d,c=decode(fitted.x)
        axial=(centers-c)@d
        if np.mean(axial)<0:d=-d;axial=-axial
        radial=centers-c-axial[:,None]*d
        e=radial[0]/np.linalg.norm(radial[0]);f=np.cross(d,e)
        angles=np.unwrap(np.arctan2(radial@f,radial@e))
        solutions.append(dict(axis=d,corner_mm=c,radius_mm=float(fitted.x[2]),
            rms_mm=float(np.sqrt(np.mean((np.linalg.norm(radial,axis=1)-fitted.x[2])**2))),
            objective=float(fitted.cost),wrap=float(abs(angles[-1]-angles[0])/(2*np.pi)),
            axial_mm=axial,solver_success=bool(fitted.success)))
    return min(solutions,key=lambda x:x['objective'])


def run():
    rows=[]
    for name,path in [('horizontal',HORIZONTAL),('vertical_down',VERTICAL)]:
        m,d,meta,_=load_checkpoint(path,verify_hashes=False)
        model=mujoco.MjData(m);qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
        model.qpos[:7]=[0,0,0,1,0,0,0];model.qpos[qadr]=d.qpos[qadr]
        mujoco.mj_kinematics(m,model)
        points=1000*np.vstack([model.site('s_head').xpos,
            *[model.joint('joint'+str(i)).xanchor for i in range(1,21)],model.site('s_tail').xpos])
        centers=(points[:-1]+points[1:])/2
        source=fit_cylinder(points[-12:],meta['settings']['radius'])
        rot=np.zeros(9);mujoco.mju_quat2Mat(rot,d.qpos[3:7]);rot=rot.reshape(3,3)
        corner,sa,ta,r=pipe_frame(m)
        for start in [0,1]:
            for stop in range(5,11):
                result=fit_branch(centers[start:stop],source)
                error=float(np.degrees(np.arccos(np.clip(abs((rot@result['axis'])@ta),0,1))))
                corner_error=float(np.linalg.norm(rot@result['corner_mm']+1000*d.qpos[:3]-corner))
                row=dict(scene=name,start=start,stop=stop,axis_error_deg=error,
                    corner_error_mm=corner_error,**result)
                rows.append(row)
                print(name,start,stop,'axis=%.1f corner=%.1f rms=%.1f R=%.1f wrap=%.2f'%
                    (error,corner_error,result['rms_mm'],result['radius_mm'],result['wrap']),flush=True)
    write_json(OUT/'orthogonal_geometry_probe.json',dict(rows=rows,
        priors='Intersecting right-angle pipe axes, comparable module-centre coil radii.',
        source='Cylinder fit to last ten joint span; no true source-axis input.',
        limitations='Candidate windows not selected; no contact/support judgment or temporal data yet.'))


if __name__=='__main__':run()
