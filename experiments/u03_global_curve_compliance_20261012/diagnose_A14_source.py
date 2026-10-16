"""Reproducible source support/carrier and U03 axis comparisons, saved states."""
from physics import *
import json
dest=BASE/'A14_DIAGNOSIS';a=np.load(BASE/'A13/trace.npz');u=np.load(ROOT/'experiments/cmu_native_underup_20261012/U03/trace.npz')
rs=[];support=[];sensitivity=[]
for sec in [0,20,30,35,40,42,45,48,50,51,51.5,52,53.8,54,60]:
    k=int(np.argmin(abs(a['t']-sec)));j=int(np.argmin(abs(u['t']-sec)));C=a['nominal_curve_points'][k];p=u['points'][j]
    x=p[15:];y=C[15:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V;err=np.linalg.norm((p-cx)@R+cy-C,axis=1)*1000
    rs.append(dict(t=float(u['t'][j]),U03_source_fit_axis_tilt_deg=float(np.rad2deg(np.arccos(np.clip(R[0,0],-1,1)))),U03_source_X_span_mm=float(np.ptp(u['centers'][j,15:21,0])*1000),posterior_fit_rms_mm=float(np.sqrt(np.mean(err[15:]**2)))))
    c=a['centers'][k];r=np.linalg.norm(c[:,1:]-[0,1],axis=1);sg=(c[:,0]>-.2)&(c[:,0]<.95)&(r>.07)&(r<.12);cc=(C[:-1]+C[1:])/2;nr=np.linalg.norm(cc[:,1:]-[0,1],axis=1);ng=(cc[:,0]>-.2)&(cc[:,0]<.95)&(nr>.07)&(nr<.12)
    support.append(dict(t=float(a['t'][k]),actual_source_geometry_modules=np.flatnonzero(sg).tolist(),nominal_source_geometry_modules=np.flatnonzero(ng).tolist(),source_contacts=np.flatnonzero(a['forces'][k,0]>.1).tolist(),source_contact_force_N=a['forces'][k,0].tolist(),M10_actual_m=c[10].tolist(),M10_C_m=cc[10].tolist(),M12_actual_m=c[12].tolist(),M12_C_m=cc[12].tolist(),actual_front6_r_mm=(np.linalg.norm(c[:6,[0,2]]-[1.1,1],axis=1)*1000).tolist()))
    if 40<=a['t'][k]<=54.01:
        for start in [14,15,16]:
            p=a['points'][k];x=p[start:];y=C[start:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V;pred=(C-cy)@R.T+cx;res=np.linalg.norm(p-pred,axis=1)*1000
            hc=(C[0]+C[1])/2;hp=(pred[0]+pred[1])/2
            sensitivity.append(dict(t=float(a['t'][k]),fixed_node_start=start,source_fit_axis_tilt_deg=float(np.rad2deg(np.arccos(np.clip(R[0,0],-1,1)))),head_pose_Y_mm=float((hp[1]-hc[1])*1000),front_max_mm=float(res[:7].max()),posterior_rms_mm=float(np.sqrt(np.mean(res[start:]**2)))))
(dest/'U03_source_axis_comparison.json').write_text(json.dumps(dict(method='U03 posterior nodes15..21 to A13 nominal world source helix at matched t. Only source-axis direction/RMS considered, not front geometry. References are not identical wave discretizations.',samples=rs),indent=2))
(dest/'source_support_timing.json').write_text(json.dumps(support,indent=2));(dest/'posterior_fit_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
n=np.arange(14,20);basis=n*(-.18388917)+(n%2)*np.pi/2;X=np.column_stack([np.sin(basis),np.cos(basis)]);out=[]
for name,z in [('A13',a),('U03',u)]:
    for key in ['raw_command','ctrl']:
        coeff=np.linalg.lstsq(X,z[key][:,14:].T,rcond=None)[0].T;phase=np.unwrap(np.arctan2(coeff[:,1],coeff[:,0]));amp=np.linalg.norm(coeff,axis=1);res=np.sqrt(np.mean((coeff@X.T-z[key][:,14:])**2,axis=1));rws=[]
        for sec in [0,20,35,40,45,48,50,51,51.5,52,53.8,55,60]:
            k=int(np.argmin(abs(z['t']-sec)));rws.append(dict(t=float(z['t'][k]),phase_change_deg=float(np.rad2deg(phase[k]-phase[0])),amplitude_deg=float(np.rad2deg(amp[k])),fit_rms_deg=float(np.rad2deg(res[k]))))
        slopes=[]
        for lo,hi in [(1,20),(35,40),(45,50),(50,51.5),(51.5,52.5)]:
            ix=(z['t']>=lo)&(z['t']<=hi);slopes.append(dict(interval=[lo,hi],phase_rate_deg_s=float(np.polyfit(z['t'][ix],np.rad2deg(phase[ix]),1)[0])))
        out.append(dict(case=name,channel=key,samples=rws,phase_rates=slopes))
(dest/'source_carrier_fit.json').write_text(json.dumps(dict(method='Tail joints14..19 fitted to fixed initial U03 sinusoidal spatial basis ws=-.18388917, parity pi/2. Common command carrier is distinct from rod-center polar change. Fit residual reported, diagnosis only.',cases=out),indent=2))
print('saved source/carrier/axis/support diagnostics; physical steps=0')
for r in sensitivity:
    if 51.99<r['t']<52.01 or 53.79<r['t']<53.81:print(r)
