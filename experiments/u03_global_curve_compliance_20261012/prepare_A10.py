"""25ms receiving/interfaces and continuous rod-envelope bound on original pipe."""
from physics import *
from continuous_window_reference import ContinuousWindowReference,LINK
from snake_control.pipe_clearance import PipeClearance
from scipy.spatial.transform import Rotation
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=np.load(INITIAL)['qpos'];mujoco.mj_forward(m,d);ref=ContinuousWindowReference(nodes(d),d.qpos[7:]);clear=PipeClearance(m,d)
out=dict(physical_steps=0,spec=ref.spec,rod_envelope_radius_m=.029,geometry='original capped cylinders and compiled elbow mesh convex-hull planes; mesh plane distance conservative',sampling='65points along each full rod; subtract half max sample spacing for Lipschitz bound on entire line',errors=[],coarse_samples=[])
previous=None;crossings=[];worst=1.;worst_row=None;peak=0.;samples={}
def check(t):
    global worst,worst_row,peak
    q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5));p=i['reference_points'];c=i['nominal_curve_points']
    gap=clear.link_minimum(c,samples=65)-.029-np.max(np.linalg.norm(np.diff(c,axis=0),axis=1))/128
    guidegap=clear.link_minimum(p,samples=65)-.029-LINK/128
    j=int(np.argmin(gap));value=float(gap[j]);peak=max(peak,float(np.rad2deg(np.max(abs(q)))))
    row=dict(t=float(t),C_line_bound_mm=value*1000,guide_line_bound_mm=float(guidegap.min()*1000),worst_module=j,max_raw_deg=float(np.rad2deg(np.max(abs(q)))),head_m=c[0].tolist(),connector_arc_m=ref.connector_arc)
    if value<worst:worst=value;worst_row=row
    samples[round(float(t),6)]=(q.copy(),p.copy(),c.copy(),i['reference_locations'].copy(),row)
    return q,i,row
for t in np.arange(0,235.001,.5):
    try:q,i,row=check(t)
    except Exception as e:out['errors'].append(dict(t=float(t),error=str(e)));break
    roles=np.column_stack([i['reference_locations']<ref.start_arc,i['reference_locations']>ref.target_arc])
    if previous is not None:
        changed=np.argwhere(roles!=previous)
        for node,side in changed:crossings.append(dict(t=float(t),node=int(node),interface=['source','target'][side]))
    previous=roles;out['coarse_samples'].append(row)
print('coarse done minmm',worst*1000,'crossings',len(crossings),flush=True)
print('errors',out['errors'],flush=True)
times=set(np.round(np.arange(38,50.0001,.025),6))
for event in crossings:
    for t in np.arange(max(0,event['t']-1),min(235,event['t']+1)+.001,.025):times.add(round(float(t),6))
for k,t in enumerate(sorted(times)):
    try:check(t)
    except Exception as e:out['errors'].append(dict(t=float(t),error=str(e)));break
    if k%400==0:print('interfaces',k,'/',len(times),'t',t,'minmm',worst*1000,flush=True)
out.update(interface_events=crossings,critical_interval_s=[38,50],dense_times_count=len(times),worst_line_bound=worst_row,max_raw_deg=peak,geometric_bound_passed=bool(worst>=0 and not out['errors']))
n=np.arange(20);w=-.183889173151851;bb=np.column_stack([np.sin(w*n+(n%2)*np.pi/2),np.cos(w*n+(n%2)*np.pi/2)])[8:];ts=np.arange(.6,20.001,.2);qs=[]
for t in ts:q,i=ref.command(ref.feed_per_cycle*.02*(t-.5));qs.append(q[8:])
qs=np.array(qs);coef=np.linalg.lstsq(bb,qs.T,rcond=None)[0].T;phase=np.unwrap(np.arctan2(coef[:,1],coef[:,0]));out['source_mapping']=dict(rate_deg_s=float(np.rad2deg(np.polyfit(ts,phase,1)[0])),fit_rms_deg=float(np.rad2deg(np.sqrt(np.mean((coef@bb.T-qs)**2)))))
fig=plt.figure(figsize=(15,9))
for k,t in enumerate([0,20,30,38,45,60]):
    if float(t) not in samples:continue
    _,p,c,_,row=samples[float(t)];ax=fig.add_subplot(2,3,k+1,projection='3d');ax.plot(*ref.global_points().T,color='gray',alpha=.3,label='G');ax.plot(*p.T,'g.-',label='geometric family');ax.plot(*c.T,'b.-',label='integrated C');ax.set(xlim=(.4,1.4),ylim=(-.3,.8),zlim=(.5,1.4),title=f't={t}s bound={row["C_line_bound_mm"]:.2f}mm');ax.view_init(22,140)
    if k==0:ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(BASE/'A10_nominal_geometry.png',dpi=130)
tt=np.array(sorted(samples));np.savez_compressed(BASE/'A10_nominal_samples.npz',t=tt,q=np.array([samples[t][0] for t in tt]),guide=np.array([samples[t][1] for t in tt]),C=np.array([samples[t][2] for t in tt]))
(BASE/'A10_PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['coarse_samples','interface_events','spec']},indent=2))
