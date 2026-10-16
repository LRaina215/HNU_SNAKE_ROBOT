"""Saved actual states vs immutable route/window; no reference adaptation."""
from physics import *
import argparse,json,csv
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

def analyze(case):
    plan=json.loads((case/'plan.json').read_text());two=plan['parameters'].get('two_windows',False) or plan['parameters'].get('continuous_windows',False) or plan['parameters'].get('supply_windows',False) or plan['parameters'].get('handover_windows',False) or plan['parameters'].get('overlap_windows',False);z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];f=z['forces'];r=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2);span=np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1)
    front=np.all((r[:,:6]>.065)&(r[:,:6]<.13)&(c[:,:6,1]>.15)&(c[:,:6,1]<1.2),axis=1)&(f[:,1,:6].sum(1)>.2)&(span>np.pi)
    full=np.all((r>.07)&(r<.12)&(c[:,:,1]>.15)&(c[:,:,1]<1.2),axis=1)&(f[:,0].sum(1)<.1)&(f[:,1].sum(1)>.2)
    fi=stable(front,t,2);bi=stable(full,t,2);false=np.flatnonzero(~full);suffix=false[-1]+1 if len(false) else 0
    post=t[-1]-t[suffix] if suffix<len(t) else 0;advance=c[-1,:,1]-c[suffix,:,1] if suffix<len(t) else np.zeros(21);shape=cylinder(z['points'][-1]);axis_angle=float(np.rad2deg(np.arccos(np.clip(abs(shape['axis'][1]),0,1))))
    success=bool(bi is not None and post>=50 and axis_angle<5 and np.all(advance>.04) and np.max(z['depth'][:,3])==0 and np.max(z['depth'][:,2])<.003 and np.max(abs(z['qpos'][:,7:]))<np.radians(90.5) and np.max(abs(z['torque']))<=2.355+1e-6)
    events=[]
    for n in range(21):
        target=(r[:,n]>.065)&(r[:,n]<.13)&(c[:,n,1]>.15)&(c[:,n,1]<1.2)&(f[:,1,max(0,n-1):min(21,n+2)].sum(1)>.2);recv=stable(target,t,.5);src=np.flatnonzero(f[:,0,n]>.1)
        events.append(dict(module=n,first_retained_target=None if recv is None else float(t[recv]),last_source=None if not len(src) else float(t[src[-1]]),max_target_Y=float(c[:,n,1].max()),final_target_radius_m=float(r[-1,n])))
    out=dict(case=case.name,physical_full_transfer_success=success,direction_success=None,direction_requires_review=True,front_stable_time=None if fi is None else float(t[fi]),full_stable_time=None if bi is None else float(t[bi]),final_full_target_seconds=float(post),module_post_y_mm=(advance*1000).tolist(),end_axis_angle_deg=axis_angle,end_pitch_mm=float(shape['pitch']*1000),end_radial_rms_mm=float(shape['radial_rms']*1000),events=events,original_xml_equal=(case/'model.xml').read_bytes()==MODEL.read_bytes(),model_unchanged=True,high_level_feedback=bool(plan.get("high_level_feedback",False)))
    (case/'audit.json').write_text(json.dumps(out,indent=2))
    with (case/'material_events.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=events[0]);w.writeheader();w.writerows(events)
    # Diagnostic progress accumulates valid-side axial increments. Free motion
    # is omitted and confidence retained; this is not exact route registration.
    rr=plan['global_route'].get('radius_m',.0895);pp=plan['global_route'].get('pitch_m',.183);a=pp/np.sqrt((2*np.pi*rr)**2+pp**2);source_r=np.linalg.norm(c[:,:,1:]-[0,1],axis=2)
    source=(source_r>.065)&(source_r<.13)&(c[:,:,0]<.95);target=(r>.065)&(r<.13)&(c[:,:,1]>.15)&(c[:,:,1]<1.2)
    increments=np.zeros((len(t),21));valid=np.zeros((len(t),21),bool)
    for i in range(1,len(t)):
        sm=source[i]&source[i-1];tm=target[i]&target[i-1];increments[i,sm]=(c[i,sm,0]-c[i-1,sm,0])/a;increments[i,tm]=(c[i,tm,1]-c[i-1,tm,1])/a;valid[i]=sm|tm
    accumulated=np.cumsum(increments,axis=0);np.savez(case/'diagnostic_progress.npz',t=t,reference_S=z['S'],per_module_valid_axial_progress=accumulated,valid=valid,confidence=valid.mean(1))
    with (case/'material_paths.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['t','module','x','y','z','reference_x','reference_y','reference_z','source_normal','target_normal','reference_S','valid_axial_progress','progress_valid'])
        rc=(z['reference_points'][:,:-1]+z['reference_points'][:,1:])/2
        for i in range(len(t)):
            for n in range(21):w.writerow([t[i],n,*c[i,n],*rc[i,n],*f[i,:,n],z['S'][i],accumulated[i,n],valid[i,n]])
    g=np.load(case/'global_route.npz')['points'];fig=plt.figure(figsize=(15,9))
    for k,sec in enumerate([0,30,40,60,100,t[-1]]):
        i=np.argmin(abs(t-sec));ax=fig.add_subplot(2,3,k+1,projection='3d');ax.plot(*g.T,color='gray',alpha=.4,label='G global');ax.plot(*z['reference_points'][i].T,'g.-',label='geometric guide window')
        if 'nominal_curve_points' in z:
            ax.plot(*z['nominal_curve_points'][i].T,'b.-',label='nominal C released/recovered')
        elif 'curvature_field' in z:
            from snake_control.spatial_bishop_backbone import SpatialBishopBackbone
            bone=SpatialBishopBackbone(z['reference_points'][i]*1000,np.full(21,64.5));phase=z['local_material_phase'][i,0];co,sn=np.cos(phase),np.sin(phase);head_rotation=z['material_rotation'][i]@np.array([[co,sn],[-sn,co]])
            sample=bone.sample(field=z['curvature_field'][i],material_rotation=head_rotation,step_mm=4);cp=np.array(sample['points_mm'])/1000;ax.plot(*cp.T,'b-',label='nominal C from material curvature')
        ax.plot(*z['points'][i].T,'r.-',label='X actual');ax.set(xlim=(.5,1.4),ylim=(-.25,1.15),zlim=(min(.3,float(z['points'][i,:,2].min())-.04),1.4),xlabel='X',ylabel='Y',zlabel='Z',title=f'{case.name} t={t[i]:.1f}s S={z["S"][i]:.2f}');ax.view_init(22,140)
        if k==0:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(case/'global_local_actual.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for n in range(6):axs[0,0].plot(t,c[:,n,2],label=f'M{n}');axs[0,1].plot(c[:,n,1],c[:,n,2]);axs[1,0].plot(t,accumulated[:,n],alpha=.7)
    axs[0,0].axhline(1,color='gray');axs[0,0].set(xlabel='time(s)',ylabel='actual Z(m)');axs[0,0].legend(ncol=3)
    axs[0,1].axhspan(.9435,1.0565,color='gray',alpha=.2);axs[0,1].set(xlabel='actual Y(m)',ylabel='actual Z(m)',title='Fixed front material side trajectories')
    axs[1,0].plot(t,z['supply_distance_m'] if 'supply_distance_m' in z else z['S']*(.0645 if two else 1),'k',label='nominal F(S)' if 'supply_distance_m' in z else 'nominal LM*S' if two else 'reference S');axs[1,0].set(xlabel='time(s)',ylabel='progress(m)',title='Valid-side axial progress/a; free motion omitted');axs[1,0].legend()
    axs[1,1].plot(t,f[:,0].sum(1),label='source normal');axs[1,1].plot(t,f[:,1].sum(1),label='target normal');axs[1,1].set(xlabel='time(s)',ylabel='normal sum(N)');axs[1,1].legend()
    fig.tight_layout();fig.savefig(case/'material_evidence.png',dpi=140);plt.close(fig)
    print(json.dumps({k:v for k,v in out.items() if k not in ['events','module_post_y_mm']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);args=ap.parse_args();analyze(BASE/args.case)


