"""Saved actual opening/closing metrics and normal-speed comparison only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import argparse
import numpy as np
import mujoco
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from target_open_close_geometry import BASE,OLD,write
from target_open_close_interface import own_helix
from target_open_close_experiment import pipe_inside_projection
from curve_guided_geometry import MODEL


def shape_rows(z,times):
    rows=[]
    for t in times:
        i=int(abs(z['sim_time']-t).argmin());p=z['points_mm'][i]
        try:fit=own_helix(p[:6])
        except Exception as e:fit=dict(reliable=False,error=repr(e),fitted_own_radius_mm=None)
        delta=p[:8]-[1100,0,1000];xy=delta[:,[0,2]]
        angle=np.unwrap(np.arctan2(xy[:,1],xy[:,0]))
        rows.append(dict(time_s=float(z['sim_time'][i]),stage=str(z['stage'][i]),
            reference_opening_mm=float(z['opening_mm'][i]) if 'opening_mm' in z else 0.,
            feed_mm=float(z['feed_mm'][i]),own_helix_fit=fit,
            median_distance_to_known_pipe_axis_mm=float(np.median(np.linalg.norm(xy,axis=1))),
            net_azimuth_span_deg=float(np.degrees(angle[-1]-angle[0])),azimuth_range_deg=float(np.degrees(np.ptp(angle))),
            pipe_axis_inside_projected_front_polygon=pipe_inside_projection(p,float(z['feed_mm'][i])),
            target_contacts=np.flatnonzero(z['contact_force_N'][i,1]>.1).tolist(),
            source_contacts=np.flatnonzero(z['contact_force_N'][i,0]>.1).tolist(),
            target_total_force_N=float(z['contact_force_N'][i,1].sum()),
            original_front_target_force_N=float(z['contact_force_N'][i,1,:8].sum()),
            actual_station_M8_M9_y_mm=float(p[[8,9,10],1].mean())))
    return rows


def projection_panel(p,feed,width=320,height=430):
    frame=np.full((height,width,3),28,np.uint8)
    scale=.68;origin=np.array([width/2,height/2+20])
    def pixel(xz):return tuple(np.rint(origin+np.array([xz[0],-xz[1]])*scale).astype(int))
    cv2.circle(frame,tuple(origin.astype(int)),round(56.5*scale),(100,170,100),-1)
    cv2.circle(frame,tuple(origin.astype(int)),3,(255,255,255),-1)
    sigma=451.5+feed-np.arange(22)*64.5;ids=np.flatnonzero(sigma>=0)
    xy=(p[ids]-[1100,0,1000])[:,[0,2]];pixels=np.array([pixel(v) for v in xy])
    if len(pixels)>1:
        cv2.polylines(frame,[pixels],False,(30,195,250),2)
        cv2.line(frame,tuple(pixels[0]),tuple(pixels[-1]),(95,95,95),1)
    for j,xy in enumerate(pixels):
        cv2.circle(frame,tuple(xy),3,(255,210,20),-1)
        if j in (0,len(pixels)-1):cv2.putText(frame,'N'+str(ids[j]),tuple(xy+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.putText(frame,'Actual front, pipe-axis view',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1)
    inside=pipe_inside_projection(p,feed)
    cv2.putText(frame,'Pipe in projected region: '+('YES' if inside else 'NO'),(8,height-32),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
    cv2.putText(frame,'Grey chord: diagnostic only',(8,height-12),cv2.FONT_HERSHEY_SIMPLEX,.42,(180,180,180),1)
    return frame


def render_video(case,rows):
    z=np.load(case/'trace.npz');old=np.load(OLD/'R2_SPEED3_B/trace.npz')
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=480,width=720);camera=mujoco.MjvCamera()
    camera.lookat[:]=[1.08,.13,.98];camera.distance=1.5;camera.azimuth=135;camera.elevation=-25
    solo=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1040,600))
    compare=cv2.VideoWriter(str(case/'comparison_baseline_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1440,650))
    if not solo.isOpened() or not compare.isOpened():raise RuntimeError('Video writer unavailable')
    stage_starts={}
    for i in range(0,len(z['sim_time']),5):
        t=float(z['sim_time'][i]);stage=str(z['stage'][i]);feed=float(z['feed_mm'][i]);R=95+float(z['opening_mm'][i])
        data.qpos[:]=z['qpos'][i];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
        actual=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        cv2.putText(actual,f'OPEN-CLOSE actual t={t:.2f}s {stage}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,255,255),1)
        cv2.putText(actual,f'Planned R={R:.1f} mm  d={feed:.2f} mm  speed=1x',(8,44),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        sr=rows[int(abs(np.array([r['time_s'] for r in rows])-t).argmin())]
        h=sr['own_helix_fit'];fitted='approx own R='+format(h['fitted_own_radius_mm'],'.1f')+' mm' if h['reliable'] else 'own helix fit uncertain'
        cv2.putText(actual,fitted,(8,68),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,220,250),1)
        panel=projection_panel(z['points_mm'][i],feed)
        sf=np.full((600,1040,3),20,np.uint8);sf[:480,:720]=actual;sf[:430,720:]=panel
        cv2.putText(sf,'Actual saved physics; front AD pauses for opening, rear compliance stays active',(12,510),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
        cv2.putText(sf,'Reference radius, estimated own radius and distance to pipe axis are separate quantities',(12,535),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1)
        contacts=[len(sr[k]) for k in ['source_contacts','target_contacts']]
        cv2.putText(sf,f'Contact modules source/target: {contacts[0]}/{contacts[1]}   Shape fitting is diagnosis only',(12,565),cv2.FONT_HERSHEY_SIMPLEX,.54,(240,240,240),1)
        solo.write(sf)
        oi=int(abs(old['sim_time']-min(t,float(old['sim_time'][-1]))).argmin())
        data.qpos[:]=old['qpos'][oi];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
        historical=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        cv2.putText(historical,f'HISTORICAL R2 t={old["sim_time"][oi]:.2f}s {old["stage"][oi]}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1)
        cv2.putText(historical,f'Original R=95 mm d={old["feed_mm"][oi]:.2f} mm; 1x',(8,44),cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,255),1)
        if t>old['sim_time'][-1]+1e-8:
            cv2.putText(historical,'TRACE ENDED: last saved frame held',(8,69),cv2.FONT_HERSHEY_SIMPLEX,.52,(30,160,255),1)
        cf=np.full((650,1440,3),20,np.uint8);cf[:480,:720]=actual;cf[:480,720:]=historical
        cv2.putText(cf,'LEFT: actual open-close action. RIGHT: historical baseline, both at normal speed.',(15,510),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),1)
        cv2.putText(cf,'Different schedules: side-by-side is a visual comparison, not a duration-matched controlled experiment.',(15,545),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
        cv2.putText(cf,'After historical trace ends, its final frame is explicitly held; no extra baseline physics is implied.',(15,580),cv2.FONT_HERSHEY_SIMPLEX,.60,(240,240,240),1)
        cv2.putText(cf,'Opening/closing must occur in the left actual trace; no nominal root motion is played.',(15,615),cv2.FONT_HERSHEY_SIMPLEX,.61,(240,240,240),1)
        compare.write(cf)
        if stage not in stage_starts:
            stage_starts[stage]=t;cv2.imwrite(str(case/('stage_'+stage+'.png')),sf)
        if any(abs(t-s)<.001 for s in [5,17,62,74,84]):cv2.imwrite(str(case/f'frame_{t:06.2f}.png'),sf)
        if i==5*((len(z['sim_time'])-1)//5):cv2.imwrite(str(case/'final_frame.png'),sf)
    solo.release();compare.release();renderer.close()
    write(case/'replay_metadata.json',dict(fps=20,sampled_physics_interval_s=.05,speed=1.,
        actual_last_time_s=float(z['sim_time'][-1]),initial_frame_included=True,frames=len(range(0,len(z['sim_time']),5)),
        historical_last_time_s=float(old['sim_time'][-1]),historical_after_end='explicitly labelled final-frame hold; not new physics',
        matching='time-aligned normal speed, schedules differ; visual comparison only'))


def main(version):
    case=BASE/(version+'_ACTION');z=np.load(case/'trace.npz')
    times=np.unique(np.r_[np.arange(0,z['sim_time'][-1]+.01,.5),[s for s in [0,5,10,15,17,27,62,74,84,129,139] if s<=z['sim_time'][-1]+1e-6],z['sim_time'][-1]])
    rows=shape_rows(z,times);write(case/'shape_timeline.json',rows)
    controls=json.loads((case/'control.json').read_text(encoding='utf8'))
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    initial=rows[0]['own_helix_fit'];openrows=[r for r in rows if r['stage']=='feed_open' and r['own_helix_fit'].get('reliable')]
    closedrows=[r for r in rows if r['stage'] in ('post_close_hold','feed_closed','end_hold') and r['own_helix_fit'].get('reliable')]
    opened=max(openrows,key=lambda r:r['own_helix_fit']['fitted_own_radius_mm']) if openrows else None
    openmedian=float(np.median([r['own_helix_fit']['fitted_own_radius_mm'] for r in openrows])) if openrows else None
    closedfirst=[r for r in closedrows if r['time_s']<=84.00001]
    closemedian=float(np.median([r['own_helix_fit']['fitted_own_radius_mm'] for r in closedfirst])) if closedfirst else None
    openincrease=openmedian-initial['fitted_own_radius_mm'] if openmedian is not None else None
    closedecrease=openmedian-closemedian if closemedian is not None else None
    # Describe model-based opening separately from strong actual pose/ACK evidence.
    actualopened=bool(openincrease is not None and openincrease>=20)
    actualcycle=bool(actualopened and closedecrease is not None and closedecrease>=15)
    p=z['centers_mm'];t=z['sim_time'];counts=np.count_nonzero(z['contact_force_N']>.1,axis=2)
    metrics=dict(physics_steps_added=0,actual_seconds=float(t[-1]),actual_frames=len(t),
        actual_commands=len(controls),opening_approximate_fit_pass=actualopened,
        open_close_approximate_fit_pass=actualcycle,initial_own_radius_fit=initial,
        open_interval_median_fitted_own_radius_mm=openmedian,post_close_median_fitted_own_radius_mm=closemedian,
        opening_own_radius_median_increase_mm=openincrease,closing_own_radius_median_decrease_mm=closedecrease,
        radius_fit_limitation='six-node uniform helix approximation, finite-shell discretization; linearized uncertainty not proof of exact intrinsic radius',
        after5_minimum_source_contacts=int(counts[t>=5-1e-8,0].min()),
        after5_minimum_target_contacts=int(counts[t>=5-1e-8,1].min()),
        ground_frames=int(np.count_nonzero(z['ground'])),self_contact_frames=int(np.count_nonzero(z['self_contact'])),
        external_force_maximum=float(z['external_applied_force_maximum'].max()),
        generalized_force_maximum=float(z['generalized_applied_force_maximum'].max()),
        maximum_actual_torque_Nm=float(abs(z['torque']).max()),
        geometric_centroid_net_displacement_mm=float(np.linalg.norm(p[-1].mean(0)-p[0].mean(0))),
        stage_counts={str(s):int(np.count_nonzero(z['stage']==s)) for s in np.unique(z['stage'])},
        conclusion='Shape occurrence and stable grasp are independent; legacy material stall remains a recorded failure',
        actual_result=result)
    write(case/'diagnostic_metrics.json',metrics)
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    ts=np.array([r['time_s'] for r in rows]);rs=np.array([r['own_helix_fit'].get('fitted_own_radius_mm',np.nan) if r['own_helix_fit'].get('reliable') else np.nan for r in rows])
    axes[0,0].plot(t,z['reference_plateau_radius_mm'],label='planned target plateau R');axes[0,0].plot(ts,rs,label='own spatial helix fit');axes[0,0].legend();axes[0,0].set_ylabel('Radius (mm), different definitions')
    axes[0,1].plot(ts,[r['own_helix_fit'].get('axis_offset_at_front_mean_y_mm',np.nan) for r in rows],label='own axis offset');axes[0,1].plot(ts,[r['median_distance_to_known_pipe_axis_mm'] for r in rows],label='distance to pipe axis');axes[0,1].legend();axes[0,1].set_ylabel('Offset / distance (mm)')
    axes[1,0].plot(t,counts[:,0],label='source');axes[1,0].plot(t,counts[:,1],label='target');axes[1,0].legend();axes[1,0].set_ylabel('Actual contact modules >0.1N')
    axes[1,1].plot(ts,[r['target_total_force_N'] for r in rows],label='all target');axes[1,1].plot(ts,[r['original_front_target_force_N'] for r in rows],label='M0..7 target');axes[1,1].legend();axes[1,1].set_ylabel('Actual normal force (N)')
    ct=[c['time_s'] for c in controls];hg=np.degrees(np.array([c['retained_shape_rad'] for c in controls]));axes[2,0].plot(ct,abs(hg).max(1),label='ACK residual');axes[2,0].plot(t,np.degrees(abs(z['q']-z['ctrl'])).max(1),label='tracking');axes[2,0].legend();axes[2,0].set_ylabel('Angle (deg)')
    axes[2,1].plot(ts,[r['actual_station_M8_M9_y_mm'] for r in rows],label='actual station');axes[2,1].plot(t,z['feed_mm'],label='planned d');axes[2,1].legend();axes[2,1].set_ylabel('Distinct coordinates (mm)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        for s in [5,17,62,74,84,129]:
            if s<=t[-1]:ax.axvline(s,color='.6',ls=':',lw=.8)
    axes[2,0].set_xlabel('Actual time (s)');axes[2,1].set_xlabel('Actual time (s)')
    fig.suptitle(version+': actual saved opening/closing diagnostics');fig.tight_layout();fig.savefig(case/'actual_diagnostics.png',dpi=140);plt.close(fig)
    render_video(case,rows)
    # Historical shape diagnosis is read-only; no fake matching duration.
    if not (BASE/'historical_shape_diagnosis.json').exists():
        with np.load(OLD/'R2_SPEED3_B/trace.npz') as old:
            write(BASE/'historical_shape_diagnosis.json',dict(physics_steps_added=0,
                rows=shape_rows(old,[0,5,34.9,50,56.34,61.8,66.8]),
                distinction='own fit radius/axis, distance to pipe axis, physical contact are separate; cannot assign unique cause'))
    print(json.dumps({k:metrics[k] for k in ['actual_seconds','opening_approximate_fit_pass','open_close_approximate_fit_pass','open_interval_median_fitted_own_radius_mm','post_close_median_fitted_own_radius_mm','after5_minimum_source_contacts']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);main(parser.parse_args().version)
