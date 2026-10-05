"""Test ordered helix registration, retaining axial progression information.

Cylinder residuals discard order/axial spacing; this diagnostic adds the
commanded pitch and initially inferred radius. It must tolerate actual
compliant deformation before it can be used as a geometry observation.
"""
import json
import numpy as np
import mujoco
from segmented_pole_experiment import ROOT,pipe_frame,write_json
from three_segment_review import context


def fitted_helix(centers,arc,radius,pitch,prior):
    h=pitch/(2*np.pi);theta=arc/np.hypot(radius,h);fits=[]
    for sign in (-1.,1.):
        template=np.column_stack((radius*np.cos(theta),sign*radius*np.sin(theta),-h*theta))
        x=template-template.mean(axis=0);y=centers-centers.mean(axis=0)
        u,_,vt=np.linalg.svd(x.T@y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@fix@u.T;shift=centers.mean(axis=0)-rotation@template.mean(axis=0)
        error=template@rotation.T+shift-centers
        fits.append(dict(axis=rotation[:,2],center=shift,handedness=sign,
            rms_mm=float(np.sqrt(np.mean(np.sum(error*error,axis=1)))),
            compatible_sign=bool(rotation[:,2]@prior>0)))
    return min(fits,key=lambda r:r['rms_mm'])


def run():
    root=ROOT/'experiments/segmented_pole_20261005/feedback_takeover';reports=[]
    for name in ('horizontal_encoder_live_whole_coil','vertical_encoder_live_whole_coil','horizontal_encoder_distinct'):
        folder=root/name;_,z,m,_,meta=context(folder)
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        radius=controls[0]['geometry_observer']['estimate']['pipe_radius_mm']+29.
        own=mujoco.MjData(m);own.qpos[:7]=[0,0,0,1,0,0,0]
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];times=z['time']-z['time'][0]
        true_corner,_,true_axis,_=pipe_frame(m);rows=[]
        for at in np.arange(0,100.1,2.):
            ci=int(np.argmin([abs(r['time']-.2-at) for r in controls]));info=controls[max(0,ci-1)]
            i=int(np.argmin(abs(times-at)));q=z['qpos'][i];own.qpos[qadr]=q[qadr];mujoco.mj_kinematics(m,own)
            p=1000*np.vstack([own.site('s_head').xpos,*[own.joint('joint'+str(j)).xanchor for j in range(1,21)],own.site('s_tail').xpos])
            centers=(p[:-1]+p[1:])/2;nodes=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];arc=(nodes[1:]+nodes[:-1])/2
            prior=info['geometry_observer']['estimate']['target'];fits=[]
            for start in (0,2):
                for end in range(7,22):
                    fit=fitted_helix(centers[start:end],arc[start:end],radius,meta['settings']['pitch'],prior)
                    fit.update(start=start,end=end,count=end-start);fits.append(fit)
            admissible=[r for r in fits if r['rms_mm']<=8. and r['compatible_sign']]
            chosen=min(admissible,key=lambda r:(-r['count'],r['rms_mm'])) if admissible else None
            rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q[3:7]);rot=rot.reshape(3,3)
            for fit in fits:
                fit['target_error_deg']=float(np.degrees(np.arccos(np.clip((rot@fit['axis'])@true_axis,-1,1))))
                fit['line_offset_mm']=float(np.linalg.norm(np.cross(rot@fit['center']+1000*q[:3]-true_corner,true_axis)))
            rows.append(dict(time=float(times[i]),chosen=chosen,fits=fits))
        valid=[r['chosen'] for r in rows if r['chosen'] is not None]
        print(name,'valid',len(valid),'median/max angle',np.median([r['target_error_deg'] for r in valid]),max(r['target_error_deg'] for r in valid),flush=True)
        print([(round(r['time']),None if r['chosen'] is None else [r['chosen']['count'],round(r['chosen']['rms_mm'],2),round(r['chosen']['target_error_deg'],2)]) for r in rows[::8]],flush=True)
        reports.append(dict(name=name,rows=rows))
    write_json(root/'ordered_helix_probe.json',dict(runs=reports,limitation='Commanded pitch and initial radius are priors, not measured pipe pose. Longest prefix under 8 mm RMS selected before truth scoring; not a control rollout.'))


if __name__=='__main__':run()
