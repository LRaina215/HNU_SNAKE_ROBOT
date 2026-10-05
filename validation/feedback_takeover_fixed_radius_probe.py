"""Assess fixed-radius local cylinder fits without an acquired-prefix input.

The radius comes from the initial inferred pipe, plus the known robot shell.
Candidate prefixes are compared by held-out body geometry, never pipe truth.
"""
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from segmented_pole_experiment import ROOT, pipe_frame, write_json
from three_segment_review import context


def fit_line(centers, radius, prior):
    mean = centers.mean(axis=0)
    p = centers - mean
    seed = np.asarray(prior)
    h = np.eye(3)[np.argmin(abs(seed))]
    h -= (h @ seed) * seed
    h /= np.linalg.norm(h)
    v = np.cross(seed, h)
    def decode(x):
        axis = seed + x[2] * h + x[3] * v
        axis /= np.linalg.norm(axis)
        e = h - (h @ axis) * axis
        e /= np.linalg.norm(e)
        return mean + x[0] * e + x[1] * np.cross(axis, e), axis
    def residual(x):
        c, a = decode(x)
        return np.linalg.norm(np.cross(centers-c, a), axis=1) - radius
    candidates = []
    # The circle may lie on either side of a partial arc. Multiple starts
    # expose this ambiguity rather than seed with a true axis or centre.
    for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
        fit = least_squares(residual, [radius*np.cos(phi), radius*np.sin(phi), 0., 0.],
            bounds=([-2*radius, -2*radius, -2., -2.], [2*radius, 2*radius, 2., 2.]),
            x_scale=[radius, radius, .3, .3], loss='soft_l1', f_scale=3., max_nfev=80)
        c, a = decode(fit.x)
        candidates.append(dict(center=c, axis=a, rms_mm=float(np.sqrt(np.mean(residual(fit.x)**2))), cost=float(fit.cost)))
    return min(candidates, key=lambda r:r['cost'])


def run():
    root = ROOT/'experiments/segmented_pole_20261005/feedback_takeover'
    reports = []
    for name in ('horizontal_encoder_live_whole_coil', 'vertical_encoder_live_whole_coil', 'horizontal_encoder_distinct'):
        folder = root/name
        _,z,m,_,_=context(folder)
        controls=json.loads((folder/'controls.json').read_text(encoding='utf8'))
        initial=controls[0]['geometry_observer']['estimate']
        radius=initial['pipe_radius_mm']+29.
        own=mujoco.MjData(m);own.qpos[:7]=[0,0,0,1,0,0,0]
        qadr=m.jnt_qposadr[m.actuator_trnid[:,0]];times=z['time']-z['time'][0]
        true_corner,_,true_axis,_=pipe_frame(m);rows=[]
        for at in (0,16,32,40,48,56,64,72,80,88,96):
            ci=int(np.argmin([abs(r['time']-.2-at) for r in controls]));info=controls[max(0,ci-1)]
            i=int(np.argmin(abs(times-at)));q=z['qpos'][i]
            own.qpos[qadr]=q[qadr];mujoco.mj_kinematics(m,own)
            points=1000*np.vstack([own.site('s_head').xpos,*[own.joint('joint'+str(j)).xanchor for j in range(1,21)],own.site('s_tail').xpos])
            centers=(points[:-1]+points[1:])/2;prior=info['geometry_observer']['estimate']['target']
            fits=[]
            for start in (0,2):
                for end in range(7,22,2):
                    fit=fit_line(centers[start:end],radius,prior)
                    distances=np.linalg.norm(np.cross(centers-fit['center'],fit['axis']),axis=1)
                    # A radius-compatible consecutive prefix, excluding the
                    # head's two known free-end links from evidence selection.
                    compatible=abs(distances-radius)<8.
                    count=2
                    while count<21 and compatible[count]:count+=1
                    error=float(np.mean(np.minimum((distances[2:count]-radius)**2,64.))) if count>2 else 64.
                    fit.update(start=start,end=end,compatible_prefix=count,
                        score=error+8*(21-count))
                    fits.append(fit)
            chosen=min(fits,key=lambda r:r['score'])
            # Read simulator pose only after every candidate and selection.
            rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q[3:7]);rot=rot.reshape(3,3)
            for fit in fits:
                fit['target_error_deg']=float(np.degrees(np.arccos(np.clip((rot@fit['axis'])@true_axis,-1,1))))
                fit['line_offset_mm']=float(np.linalg.norm(np.cross(rot@fit['center']+1000*q[:3]-true_corner,true_axis)))
            row=dict(time=float(times[i]),inferred_prefix=info['actual_prefix'],fits=fits,chosen=chosen);rows.append(row)
            print(name,at,'chosen',chosen['start'],chosen['end'],'prefix',chosen['compatible_prefix'],'axis',round(chosen['target_error_deg'],2),'offset',round(chosen['line_offset_mm'],2),'rms',round(chosen['rms_mm'],2),flush=True)
        reports.append(dict(run=name,radius_prior_mm=radius,rows=rows))
        write_json(root/'fixed_radius_probe.json',dict(runs=reports,limitation='Stateless diagnostic on three previous rollouts; no physical feedback change or support proof.'))


if __name__=='__main__':run()
