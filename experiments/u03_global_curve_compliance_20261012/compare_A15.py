"""Frozen A15 mechanism criteria and original actual acceptance, no dynamics."""
from physics import *
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stable(mask,t,seconds=2):
    start=None
    for k,v in enumerate(mask):
        if not v:start=None
        elif start is None:start=k
        elif t[k]-t[start]>=seconds:return float(t[start])
    return None

anchor=np.load(BASE/'A14/trace.npz');cases=[];data={};sensitivity=[]
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A14',BASE/'A14'),('A15',BASE/'A15')]:
    z=np.load(path/'trace.npz');t=z['t'];c=z['centers'];p=z['points'];f=z['forces'];rad=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2);yr=(c[:,:,1]>.15)&(c[:,:,1]<1.2)
    fg=(rad[:,:6]>.065)&(rad[:,:6]<.13)&yr[:,:6];span=np.rad2deg(np.ptp(np.unwrap(np.arctan2(c[:,:6,2]-1,c[:,:6,0]-1.1),axis=1),axis=1));ff=f[:,1,:6].sum(1)
    front=fg.all(1)&(span>180)&(ff>.2);tg=(rad>.07)&(rad<.12)&yr;sn=f[:,0].sum(1);full=tg.all(1)&(sn<.1)&(f[:,1].sum(1)>.2)
    sr=np.linalg.norm(c[:,:,1:]-[0,1],axis=2);sg=(c[:,:,0]>-.2)&(c[:,:,0]<.95)&(sr>.07)&(sr<.12)
    ph=np.unwrap(np.arctan2(c[:,15:21,2]-1,c[:,15:21,1]),axis=0);dx=(c[:,15:21,0]-c[0,15:21,0]).mean(1)*1000;dp=np.rad2deg((ph-ph[0]).mean(1));tilt=np.full(len(t),np.nan);poses=np.full((len(t),3),np.nan);residual=np.full(len(t),np.nan);M10err=np.full(len(t),np.nan)
    for k in np.flatnonzero(t<=60.01):
        C=z['nominal_curve_points'][k] if name!='U03' else anchor['nominal_curve_points'][int(np.argmin(abs(anchor['t']-t[k])))];x=p[k,15:];y=C[15:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V;pred=(C-cy)@R.T+cx
        tilt[k]=np.rad2deg(np.arccos(np.clip(R[0,0],-1,1)));poses[k]=((pred[0]+pred[1]-C[0]-C[1])/2)*1000;residual[k]=np.sqrt(np.mean(np.sum((p[k,15:]-pred[15:])**2,axis=1)))*1000
        if name!='U03':M10err[k]=np.linalg.norm(c[k,10]-(C[10]+C[11])/2)*1000
    active=np.flatnonzero(sn>.1);geo=np.flatnonzero(sg.any(1));false=np.flatnonzero(~full);suffix=int(false[-1]+1) if len(false) else 0;ft=stable(front,t);bt=stable(full,t)
    acquired=np.flatnonzero(fg.all(1));window=(t>=39.99)&(t<=54.01);rollback=(t>=49.99)&(t<=54.01)
    i15=int(np.argmin(abs(t-51.5)));i52=int(np.argmin(abs(t-52)));m10=(f[:,0,10]>.1);samples=[]
    for start in [14,16]:
        ss=[]
        for k in np.flatnonzero(window):
            C=z['nominal_curve_points'][k] if name!='U03' else anchor['nominal_curve_points'][int(np.argmin(abs(anchor['t']-t[k])))];x=p[k,start:];y=C[start:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V;pred=(C-cy)@R.T+cx
            ss.append(dict(t=float(t[k]),axis_tilt_deg=float(np.rad2deg(np.arccos(np.clip(R[0,0],-1,1)))),head_pose_Y_mm=float((pred[0,1]+pred[1,1]-C[0,1]-C[1,1])*500),posterior_rms_mm=float(np.sqrt(np.mean(np.sum((p[k,start:]-pred[start:])**2,axis=1)))*1000)))
        sensitivity.append(dict(case=name,fixed_node_start=start,axis_tilt_mean_deg=float(np.mean([x['axis_tilt_deg'] for x in ss])) if ss else None,axis_tilt_max_deg=float(max(x['axis_tilt_deg'] for x in ss)) if ss else None,samples=ss))
    for sec in [35,40,45,48,50,50.5,51,51.5,52,53.8,54,55,60,80,133.2,float(t[-1])]:
        if sec>t[-1]+.01:continue
        k=int(np.argmin(abs(t-sec)));C=z['nominal_curve_points'][k] if name!='U03' else None
        samples.append(dict(t=float(t[k]),source_fit_axis_tilt_deg=float(tilt[k]) if np.isfinite(tilt[k]) else None,posterior_fit_rms_mm=float(residual[k]) if np.isfinite(residual[k]) else None,M10_source_N=float(f[k,0,10]),M10_actual_m=c[k,10].tolist(),M10_C_m=((C[10]+C[11])/2).tolist() if C is not None else None,M10_world_error_mm=float(M10err[k]) if np.isfinite(M10err[k]) else None,source_geometry_modules=np.flatnonzero(sg[k]).tolist(),source_contact_modules=np.flatnonzero(f[k,0]>.1).tolist(),front_geo=fg[k].tolist(),front_Y_m=c[k,:6,1].tolist(),front_radius_mm=(rad[k,:6]*1000).tolist(),front_span_deg=float(span[k]),front_target_N=float(ff[k]),head_center_m=c[k,0].tolist(),C_head_center_m=((C[0]+C[1])/2).tolist() if C is not None else None,head_world_error_mm=((c[k,0]-(C[0]+C[1])/2)*1000).tolist() if C is not None else None,head_pose_mm=poses[k].tolist() if np.isfinite(poses[k]).all() else None,source_delta_X_mm=float(dx[k]),source_delta_polar_deg=float(dp[k]),tracking_max_deg=float(np.rad2deg(abs(z['ctrl'][k]-z['qpos'][k,7:]).max())),source_N=float(sn[k]),target_straight_count=int(tg[k].sum())))
    contacts=[]
    for v in range(6):
        ix=np.flatnonzero(f[:,1,v]>.1);contacts.append(dict(module=v,first_s=float(t[ix[0]]) if ix.size else None,last_s=float(t[ix[-1]]) if ix.size else None))
    row=dict(case=name,front_stable_s=ft,full_stable_s=bt,post_full_suffix_s=float(t[-1]-t[suffix]) if suffix<len(t) else 0,max_target_straight_rods=int(tg.sum(1).max()),max_span_when_all6_geometry_deg=float(span[acquired].max()) if acquired.size else None,any_front6_geometry_halfturn_contact=bool(front.any()),last_source_contact_s=float(t[active[-1]]) if active.size else None,last_source_straight_geometry_s=float(t[geo[-1]]) if geo.size else None,last_source_N=float(sn[-1]),source_axis_tilt_40_to_54_mean_deg=float(np.nanmean(tilt[window])) if window.any() else None,source_axis_tilt_40_to_54_max_deg=float(np.nanmax(tilt[window])) if window.any() else None,M10_source_contact_sample_s_40_to_54=float(m10[window].sum()*.1),M10_mean_source_N_40_to_54=float(f[window,0,10].mean()),M10_mean_world_error_mm_40_to_54=float(np.nanmean(M10err[window])) if name!='U03' else None,fixed_source_delta_X_51p5_to_52_mm=float(dx[i52]-dx[i15]),fixed_source_delta_polar_51p5_to_52_deg=float(dp[i52]-dp[i15]),fixed_source_max_X_drawdown_50_to_54_mm=float(np.max(np.maximum.accumulate(dx[rollback])-dx[rollback])) if rollback.any() else None,front_contacts=contacts,samples=samples)
    row['critical_window_complete']=bool(t[-1]>=54)
    row['last_saved_t']=float(t[-1])
    if t[-1]<54:
        for key in ['source_axis_tilt_40_to_54_mean_deg','source_axis_tilt_40_to_54_max_deg','M10_source_contact_sample_s_40_to_54','M10_mean_source_N_40_to_54','M10_mean_world_error_mm_40_to_54','fixed_source_max_X_drawdown_50_to_54_mm']:row[key]=None
    if t[-1]<52:
        row['fixed_source_delta_X_51p5_to_52_mm']=None;row['fixed_source_delta_polar_51p5_to_52_deg']=None
    cases.append(row);data[name]=dict(t=t,tilt=tilt,M10_source_N=f[:,0,10],dx=dx,span=span,minY=c[:,:6,1].min(1),source_N=sn)

a13=next(x for x in cases if x['case']=='A14');a14=next(x for x in cases if x['case']=='A15');keys=['source_axis_tilt_40_to_54_mean_deg','source_axis_tilt_40_to_54_max_deg','M10_mean_source_N_40_to_54','M10_mean_world_error_mm_40_to_54','fixed_source_delta_X_51p5_to_52_mm','fixed_source_max_X_drawdown_50_to_54_mm']
audit=json.loads((BASE/'A15/audit.json').read_text()) if (BASE/'A15/audit.json').exists() else {}
out=dict(physical_steps=0,criteria_frozen_before_trial='A15_DESIGN.md',method='fixed nodes15..21 actual-world to C diagnostic only, no native root rotation comparison. U03 uses A14 world source helix at matched t only for source-axis direction, not front shape. Contact sample duration is sample count*.1, not exact continuous duration. Source contact classification includes original elbow.',candidate_physical_full_transfer_success=audit.get('physical_full_transfer_success'),stage_A_independent_review_pending=True,stage_A_passed=False,stage_B_started=False,cases=cases,A15_minus_A14={key:a14[key]-a13[key] if a14[key] is not None and a13[key] is not None else None for key in keys})
(BASE/'A15_COMPARISON.json').write_text(json.dumps(out,indent=2));(BASE/'A15/comparison.json').write_text(json.dumps(out,indent=2))
(BASE/'A15/posterior_fit_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
fig,axes=plt.subplots(3,2,figsize=(13,11))
for name,dd in data.items():
    for ax,key in zip(axes.flat,['tilt','M10_source_N','dx','span','minY','source_N']):ax.plot(dd['t'],dd[key],label=name)
for ax,yl in zip(axes.flat,['source fit axis tilt (deg)','M10 source normal (N)','M15..20 X advance (mm)','front6 span (deg)','min front6 Y (m)','source normal sum (N)']):ax.set_ylabel(yl);ax.grid(alpha=.25);ax.legend()
for ax in axes.flat[:3]:ax.set_xlim(30,60);ax.axvspan(50,54,color='gray',alpha=.12)
for ax in axes.flat[3:]:ax.set_xlim(0,235)
axes[1,1].axhline(180,color='gray',ls='--');axes[2,0].axhline(.15,color='gray',ls='--')
for ax in axes[2]:ax.set_xlabel('saved actual time (s)')
fig.suptitle('A15 receiving contact contrast: source cost and actual acquisition');fig.tight_layout();fig.savefig(BASE/'A15/mechanism_comparison.png',dpi=150)
for row in cases:print({k:v for k,v in row.items() if k not in ['samples','front_contacts']},flush=True)
