"""Scoring-only straight-axis mechanism replay from saved robot qpos.

Partitions the 21 actual links once across pipe regions, integrates signed
azimuth changes, and reconstructs geometric contact locations. No simulated
contact forces are inferred, no pipe truth is supplied to a controller, and no
new physical step or EXE process is started. Geometry sampling is 5 Hz; the
source 10ms raw trajectory is retained. Supports the four prescribed scenes.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from pitch_adaptation_experiment_v5 import ROOT,SCENES,model_path
from local_pitch_feedback import estimate_module_radius

BASE=ROOT/'experiments/pitch_adaptation_20261003/repair_v5'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def truth_frame(model,scene):
    try:
        line=model.numeric('pipe_centerline').data.reshape(-1,4)[:,:3]
        origin=line[0];axis=line[-1]-origin;axis/=np.linalg.norm(axis)
        if np.max(np.linalg.norm(np.cross(line-origin,axis),axis=1))>1e-8:
            raise ValueError('This mechanism replay is restricted to straight axes')
    except KeyError:
        origin=np.zeros(3);axis=np.array([0.,0.,1.])
    reference=np.eye(3)[np.argmin(abs(axis))]
    horizontal=reference-axis*np.dot(reference,axis);horizontal/=np.linalg.norm(horizontal)
    vertical=np.cross(axis,horizontal)
    if scene.startswith('pipe_straight'):
        boundaries=np.array([]);regions=['pipe']
    else:
        try:boundaries=model.numeric('feature_span').data[:2].copy()
        except KeyError:boundaries=model.numeric('pipe_profile').data[2:4].copy()
        if len(boundaries)!=2 or boundaries[1]<=boundaries[0]:raise ValueError('Taper region metadata invalid')
        regions=['upstream','transition','downstream']
    return origin,axis,horizontal,vertical,boundaries,regions


def partition_body(points,frame):
    origin,axis,horizontal,vertical,boundaries,regions=frame
    p=np.asarray(points)-origin
    axial=p@axis;xy=p@np.column_stack((horizontal,vertical))
    lengths=np.zeros(len(regions));turns=np.zeros(len(regions));coordinates=[[] for _ in regions]
    # Split each physical link at every crossed region boundary. Signed polar
    # increments cancel direction reversals; no sum of absolute arc turns.
    for index,(start,end) in enumerate(zip(p[:-1],p[1:])):
        delta=axial[index+1]-axial[index];cuts=[0.,1.]
        if abs(delta)>1e-12:
            cuts.extend(float(u) for u in (boundaries-axial[index])/delta if 0.<u<1.)
        cuts=sorted(cuts);link_length=float(np.linalg.norm(end-start))
        for u,v in zip(cuts[:-1],cuts[1:]):
            lo=xy[index]+u*(xy[index+1]-xy[index]);hi=xy[index]+v*(xy[index+1]-xy[index])
            if min(np.linalg.norm(lo),np.linalg.norm(hi))<1e-8:
                raise ValueError('Body passes through truth axis; polar winding is undefined')
            region=int(np.searchsorted(boundaries,axial[index]+(u+v)/2*delta,side='right'))
            lengths[region]+=(v-u)*link_length
            turns[region]+=np.arctan2(lo[0]*hi[1]-lo[1]*hi[0],np.dot(lo,hi))/(2*np.pi)
            coordinates[region].extend((axial[index]+u*delta,axial[index]+v*delta))
    total_length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    overall=np.arctan2(xy[:-1,0]*xy[1:,1]-xy[:-1,1]*xy[1:,0],np.sum(xy[:-1]*xy[1:],axis=1)).sum()/(2*np.pi)
    if abs(lengths.sum()-total_length)>1e-10 or abs(turns.sum()-overall)>1e-10:
        raise ValueError('Region partition double-counted length or winding')
    centres=(points[:-1]+points[1:])/2-origin
    radial=np.linalg.norm(centres@np.column_stack((horizontal,vertical)),axis=1)
    coverage=np.array([max(x)-min(x) if x else 0. for x in coordinates])
    return lengths,turns,coverage,radial


def plot(out,result,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=arrays['sim_time'];field_time=arrays['field_time'];regions=arrays['regions']
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    axes[0,0].stackplot(times,arrays['body_length_m'].T,labels=regions,alpha=.8)
    axes[0,0].set(ylabel='Partitioned actual body length (m)',title='Each link allocated once');axes[0,0].legend(fontsize=8)
    for i,region in enumerate(regions):
        axes[0,1].plot(times,abs(arrays['signed_turns'][:,i]),label=region)
        axes[1,0].step(times,arrays['geometric_contacts'][:,i],where='post',label=region)
    axes[0,1].set(ylabel='Net azimuth turns',title='Signed increments cancel reversals')
    axes[1,0].set(ylabel='Geometric contact points',title='Position replay; no force inference')
    image=axes[1,1].pcolormesh(times,np.arange(1,22),arrays['contact_modules'].any(axis=1).T,shading='nearest',vmin=0,vmax=1,cmap='Greys')
    axes[1,1].set(ylabel='Module index',title='Modules with geometric pipe contact')
    if len(field_time):
        a=axes[2,0].pcolormesh(field_time,np.arange(22),np.degrees(arrays['alpha_rad']).T,shading='nearest',vmin=8,vmax=30,cmap='viridis')
        p=axes[2,1].pcolormesh(field_time,np.arange(22),arrays['pitch_mm'].T,shading='nearest',cmap='viridis')
        fig.colorbar(a,ax=axes[2,0],label='Controller alpha (deg)');fig.colorbar(p,ax=axes[2,1],label='Implied P (mm)')
    axes[2,0].set(ylabel='Backbone knot',title='Controller field; separate from actual shape')
    axes[2,1].set(ylabel='Backbone knot',title='P = 2 pi R_used tan(alpha)')
    for ax in axes.flat:ax.set_xlabel('Physical time (s)')
    stage='FORMAL TRAJECTORY' if result['seed']>=101 else 'TRAINING DIAGNOSTIC'
    fig.suptitle(f"{stage}: {result['scene']} / {result['method']} / seed {result['seed']} / {result['outcome']}\nNo method-effect conclusion from this single trajectory",fontsize=12)
    fig.savefig(out/'mechanism.png',dpi=150);plt.close(fig)


def replay(case,out):
    result=json.loads((case/'result.json').read_text(encoding='utf8'));scene=result['scene']
    if scene not in SCENES or result.get('controller_version')!=5:raise ValueError('Not a v5 prescribed-scene case')
    if mujoco.__version__!='3.3.0':raise ValueError('Replay requires MuJoCo 3.3.0')
    model=mujoco.MjModel.from_xml_path(str(model_path(scene)));frame=truth_frame(model,scene)
    if model.nu!=20:raise ValueError('This replay requires the main 20-joint model')
    with np.load(case/'trace.npz') as raw:
        time=raw['sim_time'];qpos=raw['qpos'];saved_contacts=raw['pipe_contacts'];phase=raw['roll_phase']
    unique_t,indices=np.unique(time,return_index=True)
    selected=indices[(unique_t>=2.2-1e-8)&np.isclose((unique_t-2.2)/.2,np.round((unique_t-2.2)/.2),atol=1e-7)]
    if not len(selected):raise ValueError('No initialized 5Hz physical samples')
    if selected[-1]!=indices[-1]:selected=np.r_[selected,indices[-1]]
    pipe_geoms={i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('pipe')}
    bodies=[model.body(name).id for name in ['head']+[f'link{i}' for i in range(1,20)]+['tail']]
    body_index={body:i for i,body in enumerate(bodies)}
    data=mujoco.MjData(model);lengths=[];turns=[];coverage=[];radial=[];contacts=[];contact_modules=[];contact_mismatch=[]
    for index in selected:
        data.qpos[:]=qpos[index];mujoco.mj_fwdPosition(model,data)
        points=np.vstack([data.site('s_head').xpos,[data.joint(f'joint{i}').xanchor for i in range(1,21)],data.site('s_tail').xpos])
        body_length,body_turns,body_coverage,body_radial=partition_body(points,frame)
        if abs(body_length.sum()-1.3545)>1e-7:raise ValueError('Actual 21-link length differs from fixed model')
        counts=np.zeros(len(frame[-1]),int);modules=np.zeros((len(frame[-1]),21),bool)
        for contact in data.contact:
            pair=(int(contact.geom1),int(contact.geom2))
            if not any(geom in pipe_geoms for geom in pair):continue
            robot=pair[1] if pair[0] in pipe_geoms else pair[0]
            module=body_index.get(int(model.geom_bodyid[robot]))
            if module is None:continue
            coordinate=float(np.dot(np.asarray(contact.pos)-frame[0],frame[1]))
            region=int(np.searchsorted(frame[4],coordinate,side='right'))
            counts[region]+=1;modules[region,module]=True
        lengths.append(body_length);turns.append(body_turns);coverage.append(body_coverage);radial.append(body_radial)
        contacts.append(counts);contact_modules.append(modules);contact_mismatch.append(int(counts.sum())-int(saved_contacts[index]))
    with gzip.open(case/'control.json.gz','rt',encoding='utf8') as f:updates=json.load(f)['control']
    estimator_times=[];errors=[];point_errors=[];unmatched=[];field_time=[];alphas=[];pitches=[];worst=None
    for row in updates:
        field_time.append(row['sim_time']);alphas.append(row['alpha_rad'])
        pitches.append([np.nan if x is None else x for x in row['implied_pitch_mm']])
        clock=row['feedback_sim_time'];position=np.searchsorted(unique_t,clock)
        choices=[x for x in (position-1,position) if 0<=x<len(unique_t)]
        if not choices:unmatched.append(clock);continue
        match=min(choices,key=lambda x:abs(unique_t[x]-clock))
        if abs(unique_t[match]-clock)>1e-8:unmatched.append(clock);continue
        data.qpos[:]=qpos[indices[match]];mujoco.mj_kinematics(model,data)
        points=np.vstack([data.site('s_head').xpos,[data.joint(f'joint{i}').xanchor for i in range(1,21)],data.site('s_tail').xpos])
        centres=(points[:-1]+points[1:])/2-frame[0]
        actual=np.linalg.norm(centres@np.column_stack(frame[2:4]),axis=1)*1000
        for module_index,fit in enumerate(row['module_estimates']):
            if fit['valid']:
                estimator_times.append(clock);errors.append(fit['radius_mm']-float(actual[fit['lo']:fit['hi']].mean()))
                point_errors.append(fit['radius_mm']-float(actual[module_index]))
                if worst is None or abs(point_errors[-1])>abs(worst['pointwise_error_mm']):
                    worst=dict(feedback_sim_time=clock,trace_index=int(indices[match]),module_index=module_index,
                               feedback_fit_radius_mm=fit['radius_mm'],actual_module_radius_mm=float(actual[module_index]),
                               actual_window_mean_radius_mm=float(actual[fit['lo']:fit['hi']].mean()),
                               lo=fit['lo'],hi=fit['hi'],pointwise_error_mm=point_errors[-1])
    if worst is not None:
        data.qpos[:]=qpos[worst['trace_index']];mujoco.mj_kinematics(model,data)
        points=np.vstack([data.site('s_head').xpos,[data.joint(f'joint{i}').xanchor for i in range(1,21)],data.site('s_tail').xpos])*1000
        diagnostic_axis=dict(point_mm=(frame[0]*1000).tolist(),horizontal=frame[2].tolist(),vertical=frame[3].tolist(),direction=frame[1].tolist())
        diagnostic=estimate_module_radius(points,diagnostic_axis,7)['modules'][worst['module_index']]
        worst['true_axis_fit_valid']=diagnostic['valid'];worst['true_axis_fit_radius_mm']=diagnostic['radius_mm']
        if diagnostic['radius_mm'] is not None:
            worst['true_axis_pointwise_error_mm']=diagnostic['radius_mm']-worst['actual_module_radius_mm']
            worst['feedback_vs_true_axis_fit_difference_mm']=worst['feedback_fit_radius_mm']-diagnostic['radius_mm']
        worst['interpretation']='true-axis fit isolates a projection/reconstruction difference; residual local error includes window mixing and circle-model mismatch; diagnostic only'
    lengths=np.asarray(lengths);turns=np.asarray(turns);contacts=np.asarray(contacts);modules=np.asarray(contact_modules)
    straddling=(lengths[:,0]>0)&(lengths[:,-1]>0) if len(frame[-1])==3 else np.zeros(len(selected),bool)
    representative=int(np.argmax(np.minimum(lengths[:,0],lengths[:,-1]))) if straddling.any() else len(selected)-1
    def sample_metrics(mask):
        return {region:dict(n=int(mask.sum()),median_body_length_m=float(np.median(lengths[mask,i])) if mask.any() else None,
                            median_signed_turns=float(np.median(turns[mask,i])) if mask.any() else None,
                            median_net_wrap_turns=float(np.median(abs(turns[mask,i]))) if mask.any() else None,
                            median_geometric_contacts=float(np.median(contacts[mask,i])) if mask.any() else None,
                            median_contact_modules=float(np.median(modules[mask,i].sum(axis=1))) if mask.any() else None)
                for i,region in enumerate(frame[-1])}
    sources=[case/'result.json',case/'trace.npz',case/'control.json.gz',model_path(scene),Path(__file__),
             ROOT/'codes/local_pitch_feedback.py',ROOT/'codes/local_pitch_geometry.py']
    summary=dict(scene=scene,method=result['method'],seed=result['seed'],k=result['k'],outcome=result['outcome'],success=result['success'],
                 sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},scoring_only=True,
                 definitions=dict(length='each actual link partitioned once at truth region boundaries',
                                  winding='signed polar increments; net wrap is absolute after summing within region',
                                  contacts='geometric locations reconstructed from qpos; no force or stable-support inference',
                                  radius_error='accepted module fit minus mean ACTUAL link-centre radius in the same window, not pipe radius',
                                  sampling_s=.2),regions=frame[-1],frames=len(selected),straddling_frames=int(straddling.sum()),
                 straddling=sample_metrics(straddling),all_frames=sample_metrics(np.ones(len(selected),bool)),
                 representative_sim_time=float(time[selected[representative]]),
                 maximum_saved_contact_count_discrepancy=int(max(abs(np.asarray(contact_mismatch)))),
                 maximum_length_partition_error_m=float(max(abs(lengths.sum(axis=1)-1.3545))),
                 accepted_radius_samples=len(errors),accepted_radius_window_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                 accepted_radius_abs_error_p95_mm=float(np.percentile(abs(np.asarray(errors)),95)) if errors else None,
                 accepted_radius_pointwise_rmse_mm=float(np.sqrt(np.mean(np.square(point_errors)))) if point_errors else None,
                 accepted_radius_pointwise_abs_error_p95_mm=float(np.percentile(abs(np.asarray(point_errors)),95)) if point_errors else None,
                 accepted_radius_pointwise_abs_error_max_mm=float(max(abs(np.asarray(point_errors)))) if point_errors else None,
                 worst_accepted_pointwise_example=worst,
                 unmatched_feedback_times=unmatched)
    out.mkdir(parents=True,exist_ok=True)
    arrays=dict(sim_time=time[selected],body_length_m=lengths,signed_turns=turns,
                axial_coverage_m=np.asarray(coverage),actual_module_radius_mm=np.asarray(radial)*1000,
                geometric_contacts=contacts,contact_modules=modules,recorded_total_contacts=saved_contacts[selected],
                roll_phase=phase[selected],straddling=straddling,field_time=np.asarray(field_time),
                alpha_rad=np.asarray(alphas),pitch_mm=np.asarray(pitches),radius_error_time=np.asarray(estimator_times),
                accepted_window_radius_error_mm=np.asarray(errors),regions=np.asarray(frame[-1]))
    arrays['accepted_pointwise_radius_error_mm']=np.asarray(point_errors)
    np.savez_compressed(out/'mechanism.npz',**arrays)
    (out/'mechanism.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    plot(out,result,arrays)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();case=args.case.resolve()
    if ROOT not in case.parents:raise ValueError('Case must be within the experiment repository')
    output=args.out or BASE/'mechanism_v1'/case.name
    print(json.dumps(replay(case,output),indent=2,ensure_ascii=False))
