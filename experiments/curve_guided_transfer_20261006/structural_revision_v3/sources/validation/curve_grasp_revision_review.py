"""Re-evaluate saved V3 geometry on SO(3). No solver or physical stepping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation,Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.curve_grasp_revision_geometry import BASE,DiscreteFit,write


def main():
    with np.load(BASE/'fitted_reference.npz') as data:
        feeds,states=data['feed_mm'],data['states']
    old=json.loads((BASE/'geometry_result.json').read_text(encoding='utf8'))
    interpolate=PchipInterpolator(feeds,states,axis=0)
    orientations=Slerp(feeds,Rotation.from_rotvec(states[:,3:6]))
    model=DiscreteFit()
    rows=[]
    for feed in np.arange(0,161.25+.001,8.0625):
        x=interpolate(feed);x[3:6]=orientations(feed).as_rotvec()
        model.set(x);mujoco.mj_forward(model.m,model.d)
        contacts=[]
        for c in model.d.contact:
            b1,b2=[int(model.m.geom_bodyid[g]) for g in [c.geom1,c.geom2]]
            contacts.append(dict(body1=model.m.body(b1).name,body2=model.m.body(b2).name,
                geom1=model.m.geom(c.geom1).name,geom2=model.m.geom(c.geom2).name,
                nonadjacent_self=bool(b1>=model.m.body('head').id and b2>=model.m.body('head').id and abs(b1-b2)>1),
                dist_mm=float(c.dist*1000)))
        rows.append(dict(feed_mm=float(feed),minimum_conservative_gap_mm=float(model.gaps(x).min()),
            minimum_actual_contact_mm=min((c['dist_mm'] for c in contacts),default=0.),contacts=contacts))
    checks=np.arange(0,161.25+.001,.25)
    step=np.degrees(abs(np.diff(interpolate(checks)[:,6:],axis=0)))
    index=np.unravel_index(step.argmax(),step.shape)
    ready=bool(step.max()<=.5 and all(r['minimum_actual_contact_mm']>=0 and r['minimum_conservative_gap_mm']>=-.05 for r in rows))
    result=dict(design_status='READY_FOR_MINIMAL_SIM' if ready else 'NOT_READY',physics_steps=0,new_dynamics_trials=0,
        scope='Saved discrete poses; PCHIP translation/joints and shortest-path SLERP orientation. No body motion observed.',
        maximum_50ms_nominal_step_deg=float(step[index]),maximum_step_feed_mm=float(checks[index[0]]),maximum_step_joint=int(index[1]+1),
        fixed_speed_mm_s=5.,existing_target_speed_deg_s=10.,maximum_required_target_speed_deg_s=float(step.max()/.05),
        interpolation_checks=rows,
        largest_actual_penetration_mm=float(max(0,-min(r['minimum_actual_contact_mm'] for r in rows))),
        decision='Geometry admission only; holding is not established' if ready else 'Stop this candidate before C0: continuous geometry and command-rate checks failed. No B, F or real run.',
        notes=['The earlier 33.43mm event includes a rotation-vector interpolation artifact and is not used as the final mechanism claim.',
               'Isolated pose feasibility and near-contact surfaces do not establish holding.',
               'This does not prove no continuous solution exists or that active material feed is impossible.'])
    write('interpolation_review.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['feed_mm'] for r in rows],[r['minimum_actual_contact_mm'] for r in rows],'o-')
    axes[0].axhline(0,color='black',linewidth=.8)
    axes[0].set(xlabel='Reference material feed (mm)',ylabel='Actual collision distance (mm)',title='Offline interpolated poses, SLERP root')
    axes[1].plot(checks[:-1],step.max(axis=1))
    axes[1].axhline(.5,color='red',linestyle='--',label='10 deg/s, 50 ms')
    axes[1].set(xlabel='Reference material feed (mm)',ylabel='Maximum nominal target step (deg)',title='Fixed 5 mm/s schedule')
    axes[1].legend()
    fig.suptitle('V3 OFFLINE ONLY: independent contact-near poses do not form a validated transfer path')
    fig.tight_layout();fig.savefig(BASE/'continuity_failure.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['design_status','maximum_50ms_nominal_step_deg','maximum_step_feed_mm','maximum_step_joint','largest_actual_penetration_mm','new_dynamics_trials']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
