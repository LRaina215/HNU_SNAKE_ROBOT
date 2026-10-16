"""Fixed material source/world-pose audit of saved traces. No physical steps."""
from physics import *
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outdir=BASE/'A14_DIAGNOSIS';outdir.mkdir(exist_ok=True)
rows={};series={}
times=[0,10,20,30,35,39,40,42,45,48,49,50,50.5,51,51.5,52,53,53.8,54,55,60]
m=mujoco.MjModel.from_xml_path(str(BASE/'A13/model.xml'));qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A13',BASE/'A13')]:
    z=np.load(path/'trace.npz'); t=z['t']; c=z['centers']; p=z['points']
    phi=np.unwrap(np.arctan2(c[:,15:21,2]-1,c[:,15:21,1]),axis=0)
    phi0=phi[0]; dx=(c[:,15:21,0]-c[0,15:21,0]).mean(1)*1000
    dp=np.rad2deg((phi-phi0).mean(1))
    st=dict(t=t,actual_source_delta_X_mm=dx,actual_source_delta_polar_deg=dp,
        source_polar_spread_deg=np.rad2deg(np.std(phi-phi0,axis=1)))
    if name=='A13':
        C=z['nominal_curve_points'];cc=(C[:,:-1]+C[:,1:])/2
        cp=np.unwrap(np.arctan2(cc[:,15:21,2]-1,cc[:,15:21,1]),axis=0)
        st.update(nominal_source_delta_X_mm=(cc[:,15:21,0]-cc[0,15:21,0]).mean(1)*1000,
            nominal_source_delta_polar_deg=np.rad2deg((cp-cp[0]).mean(1)),
            source_world_polar_error_deg=np.rad2deg((phi-cp).mean(1)))
    selected=[]
    for sec in times:
        k=int(np.argmin(abs(t-sec))); front=c[k,:6]; span=np.ptp(np.unwrap(np.arctan2(front[:,2]-1,front[:,0]-1.1)))
        rad=np.linalg.norm(front[:,[0,2]]-[1.1,1],axis=1)
        srad=np.linalg.norm(c[k,15:21,1:]-[0,1],axis=1)
        row=dict(t=float(t[k]),head_center_m=c[k,0].tolist(),front_Y_m=front[:,1].tolist(),front_span_deg=float(np.rad2deg(span)),
            front_geometry=((rad>.065)&(rad<.13)&(front[:,1]>.15)&(front[:,1]<1.2)).tolist(),front_target_N=float(z['forces'][k,1,:6].sum()),
            source_N=float(z['forces'][k,0].sum()),source_contact_modules=np.flatnonzero(z['forces'][k,0]>.1).tolist(),
            source_fixed_M15_to_M20_radius_mm=(srad*1000).tolist(),source_fixed_X_m=c[k,15:21,0].tolist(),
            actual_source_delta_X_mm=float(dx[k]),actual_source_delta_polar_deg=float(dp[k]),source_polar_spread_deg=float(st['source_polar_spread_deg'][k]),
            source_fixed_per_rod_delta_polar_deg=np.rad2deg(phi[k]-phi0).tolist(),
            tracking_max_deg=float(np.rad2deg(np.max(abs(z['ctrl'][k]-z['qpos'][k,qa])))))
        if name=='A13':
            # Only actual world points and world C are registered. Never compare
            # this rotation with native FK whose root is arbitrary.
            x=p[k,15:22];y=C[k,15:22];cx=x.mean(0);cy=y.mean(0)
            u,sv,v=np.linalg.svd((x-cx).T@(y-cy));R=u@np.diag([1,1,np.linalg.det(u@v)])@v
            predicted=(C[k]-cy)@R.T+cx
            hactual=(p[k,0]+p[k,1])/2;hnom=(C[k,0]+C[k,1])/2;hpred=(predicted[0]+predicted[1])/2
            residual=np.linalg.norm(p[k]-predicted,axis=1)*1000
            row.update(S=float(z['S'][k]),F_m=float(z['supply_distance_m'][k]),M_m=float(z['source_chart_shift_m'][k]),
                common_roll_phase_rad=float(z['roll_phase'][k]),C_head_center_m=hnom.tolist(),
                head_world_error_mm=((hactual-hnom)*1000).tolist(),head_pose_component_mm=((hpred-hnom)*1000).tolist(),
                head_shape_component_mm=((hactual-hpred)*1000).tolist(),
                posterior_M15_to_M20_fit_front_max_mm=float(residual[:7].max()),posterior_fit_rms_mm=float(np.sqrt(np.mean(residual[15:]**2))),
                posterior_actual_to_worldC_rotation_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))),
                nominal_source_delta_X_mm=float(st['nominal_source_delta_X_mm'][k]),nominal_source_delta_polar_deg=float(st['nominal_source_delta_polar_deg'][k]),
                source_world_polar_error_deg=float(st['source_world_polar_error_deg'][k]),
                nominal_source_fixed_X_m=cc[k,15:21,0].tolist(),nominal_source_fixed_radius_mm=(np.linalg.norm(cc[k,15:21,1:]-[0,1],axis=1)*1000).tolist())
            # Constrained source-axis rotation is also world-to-world and needs
            # no native root. It cannot model tilt or local shape change.
            phase=float(np.arctan2(np.sin(phi[k]-cp[k]).mean(),np.cos(phi[k]-cp[k]).mean()))
            axisfit=C[k].copy();yy=axisfit[:,1].copy();zz=axisfit[:,2]-1
            axisfit[:,1]=np.cos(phase)*yy-np.sin(phase)*zz;axisfit[:,2]=1+np.sin(phase)*yy+np.cos(phase)*zz
            axisfit[:,0]+=(c[k,15:21,0]-cc[k,15:21,0]).mean()
            axishead=(axisfit[0]+axisfit[1])/2
            row.update(source_axis_rotation_phase_deg=float(np.rad2deg(phase)),source_axis_only_head_pose_mm=((axishead-hnom)*1000).tolist(),
                source_actual_per_rod_normal_N=z['forces'][k,0].tolist(),target_actual_per_rod_normal_N=z['forces'][k,1].tolist(),
                nominal_front_Y_m=cc[k,:6,1].tolist(),nominal_front_span_deg=float(np.rad2deg(np.ptp(np.unwrap(np.arctan2(cc[k,:6,2]-1,cc[k,:6,0]-1.1))))),
                posterior_world_rotation_matrix=R.tolist(),
                source_fit_axis_tilt_deg=float(np.rad2deg(np.arccos(np.clip(R[0,0],-1,1)))),
                source_actual_X_span_mm=float(np.ptp(c[k,15:21,0])*1000),source_nominal_X_span_mm=float(np.ptp(cc[k,15:21,0])*1000))
        selected.append(row)
    rows[name]=selected;series[name]=st

out=dict(method='Saved actual states only; fixed source materials M15..M20 and nodes15..21, original X-axis at Y0 Z1. Actual-world-to-C rigid fit is diagnostic, never a control input. Pose+shape head components sum exactly to world error. Fixed IDs avoid changing-mask means; source radii/X and fit residual show applicability. Common roll phase is distinct from centerline polar phase.',cases=rows)
(outdir/'fixed_material_timing.json').write_text(json.dumps(out,indent=2))
np.savez_compressed(outdir/'source_series.npz',**{name+'_'+key:val for name,vals in series.items() for key,val in vals.items()})
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
for name,st in series.items():
    ax[0].plot(st['t'],st['actual_source_delta_X_mm'],label=name+' actual M15..20')
    ax[1].plot(st['t'],st['actual_source_delta_polar_deg'],label=name+' actual M15..20')
st=series['A13'];ax[0].plot(st['t'],st['nominal_source_delta_X_mm'],'--',label='A13 C M15..20');ax[1].plot(st['t'],st['nominal_source_delta_polar_deg'],'--',label='A13 C M15..20')
rr=rows['A13'];tt=[r['t'] for r in rr]
ax[2].plot(tt,[r['head_world_error_mm'][1] for r in rr],label='head Y world error')
ax[2].plot(tt,[r['head_pose_component_mm'][1] for r in rr],label='head Y pose component')
ax[2].plot(tt,[r['head_shape_component_mm'][1] for r in rr],label='head Y residual shape')
for a in ax:a.set_xlim(0,60);a.axvspan(50,54,color='gray',alpha=.15);a.legend();a.grid(alpha=.25)
ax[0].set(ylabel='source axial advance (mm)',ylim=(-5,180));ax[1].set(ylabel='source material polar change (deg)',ylim=(-155,25));ax[2].set(ylabel='A13 head Y error (mm)',xlabel='saved actual time (s)')
fig.suptitle('Fixed material source convection and actual world pose, acquisition loss at 50..54 s');fig.tight_layout();fig.savefig(outdir/'source_pose_timing.png',dpi=150)
for name,rs in rows.items():
    print(name)
    for r in rs:
        if 39<=r['t']<=55.01:print(json.dumps({k:v for k,v in r.items() if k in ['t','S','F_m','M_m','front_span_deg','front_target_N','tracking_max_deg','actual_source_delta_X_mm','actual_source_delta_polar_deg','nominal_source_delta_X_mm','nominal_source_delta_polar_deg','head_world_error_mm','head_pose_component_mm','head_shape_component_mm','posterior_M15_to_M20_fit_front_max_mm','posterior_fit_rms_mm','source_polar_spread_deg','source_contact_modules']}))
