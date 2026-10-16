"""Audit saved actual material paths; never step dynamics or alter N10."""
from native_run import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stable(mask,t,seconds):
    start=None
    for i,v in enumerate(mask):
        if not v:start=None
        elif start is None:start=i
        elif t[i]-t[start]>=seconds:return start
    return None

def audit(case):
    z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];f=z['forces'];r=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2)
    angle=np.unwrap(np.arctan2(c[:,:,2]-1,c[:,:,0]-1.1),axis=1);span=np.ptp(angle[:,:6],axis=1)
    front=np.all((r[:,:6]>.065)&(r[:,:6]<.13)&(c[:,:6,1]>.15)&(c[:,:6,1]<1.2),axis=1)&(f[:,1,:6].sum(1)>.2)&(span>np.pi)
    full=np.all((r>.07)&(r<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2),axis=1)&(f[:,0].sum(1)<.1)&(f[:,1].sum(1)>.2)
    fi=stable(front,t,2);bi=stable(full,t,2);false=np.flatnonzero(~full);suffix=false[-1]+1 if len(false) else 0
    post=t[-1]-t[suffix] if suffix<len(t) else 0;advance=c[-1,:,1]-c[suffix,:,1] if suffix<len(t) else np.zeros(21)
    shape=cylinder(z['points'][-1]);axis_angle=np.rad2deg(np.arccos(np.clip(abs(shape['axis'][1]),0,1)))
    # Free approach is outside the near-coil band, beyond the target straight entrance.
    # These events are descriptive. Continuous side video and full material paths
    # remain the direction evidence, not an isolated height or this threshold.
    events=[]
    for n in range(21):
        near=(c[:,n,1]>.03)&(r[:,n]>.14)&(r[:,n]<.45)
        below=np.flatnonzero(near&(c[:,n,2]<1-.0565));above=np.flatnonzero(near&(c[:,n,2]>1+.0565))
        target=(r[:,n]>.065)&(r[:,n]<.13)&(c[:,n,1]>.15)&(c[:,n,1]<1.2)&(f[:,1,max(0,n-1):min(21,n+2)].sum(1)>.2)
        recv=stable(target,t,.5);si=np.flatnonzero(f[:,0,n]>.1)
        events.append(dict(module=n,first_free_lower=None if not len(below) else float(t[below[0]]),first_free_upper=None if not len(above) else float(t[above[0]]),first_retained_target=None if recv is None else float(t[recv]),last_source_contact=None if not len(si) else float(t[si[-1]])))
    physical=bool(bi is not None and post>=20 and axis_angle<5 and np.max(abs(z['qpos'][:,7:]))<np.radians(90.5) and np.max(abs(z['torque']))<=2.355+1e-6 and np.all(advance>.04) and np.max(z['depth'][:,3])==0)
    head=events[0];lower_first=bool(head['first_free_lower'] is not None and (head['first_free_upper'] is None or head['first_free_lower']<head['first_free_upper']))
    out=dict(case=case.name,physical_full_transfer_success=physical,lower_approach_before_upper=lower_first,direction_transfer_success=None,direction_review_required='Lower-first event alone cannot prove upward acquisition. Inspect identical material paths during near-pipe winding and side video.',front_stable_time=None if fi is None else float(t[fi]),full_stable_time=None if bi is None else float(t[bi]),final_full_target_seconds=float(post),mean_post_transfer_y_advance_mm=float(advance.mean()*1000),axis_angle_deg=float(axis_angle),end_pitch_mm=float(shape['pitch']*1000),end_radial_rms_mm=float(shape['radial_rms']*1000),max_actual_joint_deg=float(np.rad2deg(np.max(abs(z['qpos'][:,7:])))),max_torque_nm=float(np.max(abs(z['torque']))),max_depth_mm=(z['depth'].max(0)*1000).tolist(),peak_front_span_deg=float(np.rad2deg(span.max())),events=events,normal_force_is_not_vertical_support=True)
    (case/'audit.json').write_text(json.dumps(out,indent=2))
    with (case/'material_events.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=events[0]);w.writeheader();w.writerows(events)
    with (case/'material_paths.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['t','module','x','y','z','target_radius','source_normal_N','target_normal_N'])
        for i in range(len(t)):
            for n in range(21):w.writerow([t[i],n,*c[i,n],r[i,n],*f[i,:,n]])
    fig,axs=plt.subplots(2,2,figsize=(12,8));colors=plt.cm.tab10(np.arange(6))
    for n in range(6):
        axs[0,0].plot(c[:,n,1],c[:,n,2],color=colors[n],label=f'M{n}');axs[0,1].plot(t,c[:,n,2],color=colors[n]);axs[1,0].plot(t,r[:,n],color=colors[n])
    axs[0,0].axhspan(.9435,1.0565,color='gray',alpha=.25);axs[0,0].set(xlabel='world Y (m)',ylabel='world Z (m)',title='Actual material paths, side projection');axs[0,0].legend(ncol=3)
    axs[0,1].axhline(1,color='gray');axs[0,1].set(xlabel='time (s)',ylabel='world Z (m)',title='Front material heights')
    axs[1,0].axhspan(.065,.13,color='gray',alpha=.25);axs[1,0].set(xlabel='time (s)',ylabel='target-axis radius (m)')
    axs[1,1].plot(t,f[:,0].sum(1),label='source normal');axs[1,1].plot(t,f[:,1].sum(1),label='target normal');axs[1,1].set(xlabel='time (s)',ylabel='sum contact normal (N)');axs[1,1].legend()
    fig.suptitle(case.name+' saved actual trajectory');fig.tight_layout();fig.savefig(case/'material_evidence.png',dpi=140);plt.close(fig)
    print(json.dumps({k:v for k,v in out.items() if k!='events'},indent=2));return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();audit(BASE/a.case)
