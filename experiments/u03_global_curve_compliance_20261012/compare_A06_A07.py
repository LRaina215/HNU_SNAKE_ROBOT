"""Saved-state comparison after one front release intervention, no steps."""
from physics import *
import json
def align(x,y,ids):
    xx=x[ids];yy=y[ids];u,_,v=np.linalg.svd((xx-xx.mean(0)).T@(yy-yy.mean(0)))
    r=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T
    return x@r.T+yy.mean(0)-xx.mean(0)@r.T
out={}
for case in ['A06','A07']:
    z=np.load(BASE/case/'trace.npz');samples=[]
    for sec in [10,20,25,30,35,40,50,60]:
        i=np.argmin(abs(z['t']-sec));C=z['nominal_curve_points'][i] if case=='A07' else z['reference_points'][i]
        p=z['points'][i];c=z['centers'][i];g=z['reference_points'][i];rear=np.arange(10,22);polar=np.angle(np.exp(1j*(np.arctan2(p[rear,2]-1,p[rear,1])-np.arctan2(g[rear,2]-1,g[rear,1]))))
        samples.append(dict(t=float(z['t'][i]),M0_m=c[0].tolist(),C_head_m=C[0].tolist(),front_target_normal_N=float(z['forces'][i,1,:6].sum()),front_raw_deg=np.rad2deg(z['raw_command'][i,:6]).tolist(),front_tracking_max_deg=float(np.rad2deg(np.max(abs(z['ctrl'][i,:6]-z['qpos'][i,7:13])))),rear_X_mean_actual_minus_G_mm=float(np.mean(p[rear,0]-g[rear,0])*1000),rear_polar_mean_actual_minus_G_deg=float(np.rad2deg(np.mean(polar))),actual_to_C_all_rigid_registered_rms_mm=float(np.sqrt(np.mean(np.sum((align(p,C,np.arange(22))-C)**2,axis=1)))*1000)))
    mask=(z['t']>=.5)&(z['t']<=40)
    peak=np.unravel_index(np.argmax(abs(z['raw_command'][mask,:6])),z['raw_command'][mask,:6].shape)
    out[case]=dict(samples=samples,front_raw_peak_first40_deg=float(np.rad2deg(np.max(abs(z['raw_command'][mask,:6])))),front_raw_peak_t=float(z['t'][mask][peak[0]]),front_raw_peak_joint=int(peak[1]+1),result=json.loads((BASE/case/'result.json').read_text()))
(BASE/'A06_A07_actual_comparison.json').write_text(json.dumps(out,indent=2))
for case in out:
    print(case,'front40peak',out[case]['front_raw_peak_first40_deg'])
    for row in out[case]['samples']:print(row['t'],'M0Z',row['M0_m'][2],'frontN',row['front_target_normal_N'],'rearXmm',row['rear_X_mean_actual_minus_G_mm'],'rearPolar',row['rear_polar_mean_actual_minus_G_deg'])
