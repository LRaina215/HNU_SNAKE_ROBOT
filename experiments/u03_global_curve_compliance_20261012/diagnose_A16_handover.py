"""M0..M6 acquisition, command shape and world pose from saved traces only."""
from physics import *
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dest=BASE/'A16_DIAGNOSIS';dest.mkdir(exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);initial=np.load(INITIAL)['qpos'];qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
def place(P,source):
    x=P[15:];y=source[15:];cx=x.mean(0);cy=y.mean(0);U,s,V=np.linalg.svd((x-cx).T@(y-cy));R=U@np.diag([1,1,np.linalg.det(U@V)])@V
    return (P-cx)@R+cy
def centers(P):return (P[:-1]+P[1:])/2
def polar(c):return np.rad2deg(np.arctan2(c[:,2]-1,c[:,0]-1.1))
sets={};selected={};wave=[]
for name,path in [('U03',ROOT/'experiments/cmu_native_underup_20261012/U03'),('A14',BASE/'A14'),('A15',BASE/'A15')]:
    z=np.load(path/'trace.npz');t=z['t'];p=z['points'];c=z['centers'];f=z['forces'];ctrl=z['ctrl'];raw=z['raw_command'];qactual=z['qpos'][:,qa];C=z['nominal_curve_points'] if 'nominal_curve_points' in z else None;rows=[]
    for k in np.flatnonzero((t>=49.99)&(t<=56.41)):
        parts={}
        for label,q in [('raw',raw[k]),('ctrl',ctrl[k])]:
            d.qpos[:]=initial;d.qpos[qa]=q;mujoco.mj_forward(m,d);pred=place(nodes(d),p[k]);pc=centers(pred)
            parts[label]=dict(M0_to_M6_centers_m=pc[:7].tolist(),M0_to_M6_polar_deg=polar(pc[:7]).tolist(),M0_to_M6_r_mm=(np.linalg.norm(pc[:7,[0,2]]-[1.1,1],axis=1)*1000).tolist(),front_actual_prediction_error_mm=(np.linalg.norm(pc[:7]-c[k,:7],axis=1)*1000).tolist(),posterior_fit_rms_mm=float(np.sqrt(np.mean(np.sum((pred[15:]-p[k,15:])**2,axis=1)))*1000))
        cc=centers(C[k]) if C is not None else None
        if C is not None:
            posed=centers(place(C[k],p[k]));parts['C_in_actual_source_pose']=dict(M0_to_M6_centers_m=posed[:7].tolist(),M0_to_M6_polar_deg=polar(posed[:7]).tolist(),M0_to_M6_r_mm=(np.linalg.norm(posed[:7,[0,2]]-[1.1,1],axis=1)*1000).tolist())
        err=np.rad2deg(abs(ctrl[k]-qactual[k]));row=dict(t=float(t[k]),actual_M0_to_M6_centers_m=c[k,:7].tolist(),actual_M0_to_M6_polar_deg=polar(c[k,:7]).tolist(),actual_M0_to_M6_r_mm=(np.linalg.norm(c[k,:7][:,[0,2]]-[1.1,1],axis=1)*1000).tolist(),actual_M0_to_M6_target_N=f[k,1,:7].tolist(),actual_M8_to_M14_centers_m=c[k,8:15].tolist(),actual_M8_to_M14_source_N=f[k,0,8:15].tolist(),nominal_M0_to_M6_centers_m=cc[:7].tolist() if cc is not None else None,nominal_M0_to_M6_polar_deg=polar(cc[:7]).tolist() if cc is not None else None,nominal_M8_to_M14_centers_m=cc[8:15].tolist() if cc is not None else None,tracking_max_deg=float(err.max()),tracking_worst_joint_1based=int(err.argmax()+1),tracking_front6_max_deg=float(err[:6].max()),source_N=float(f[k,0].sum()),target_N=float(f[k,1].sum()),shape_anchored_in_actual_source=parts,raw_first6_deg=np.rad2deg(raw[k,:6]).tolist(),actual_first6_deg=np.rad2deg(qactual[k,:6]).tolist())
        if C is not None:row.update(S=float(z['S'][k]),nominal_target_node_count=int(round(float(z['target_reference'][k])*22)))
        rows.append(row)
    selected[name]=[min(rows,key=lambda r:abs(r['t']-sec)) for sec in [50,50.7,51.5,52,53,53.6,53.9,54.9,56.4]]
    sets[name]=rows
    n=np.arange(6);basis=-.18388917*n+(n%2)*np.pi/2;X=np.column_stack([np.sin(basis),np.cos(basis)]);coef=np.linalg.lstsq(X,raw[:,:6].T,rcond=None)[0].T;phase=np.unwrap(np.arctan2(coef[:,1],coef[:,0]));amp=np.linalg.norm(coef,axis=1);rss=np.sqrt(np.mean((coef@X.T-raw[:,:6])**2,axis=1))
    wave.append(dict(case=name,method='First6 raw joint command fitted to fixed U03 spatial basis. Intrinsic command phase only, not material polar phase or directly comparable arbitrary-root world rotation; partial target/connector supports and fit RMS limit interpretation.',samples=[dict(t=float(t[k]),phase_deg=float(np.rad2deg(phase[k])),amplitude_deg=float(np.rad2deg(amp[k])),fit_rms_deg=float(np.rad2deg(rss[k]))) for sec in [35,40,45,50,50.7,52,53.9,56.4,60] for k in [int(np.argmin(abs(t-sec)))]]))
out=dict(physical_steps=0,method='Fixed posterior nodes15..21 registration of native raw/ctrl FK to actual world source nodes, diagnostics only. Native root transforms are never compared directly to actual world rotations. C transplanted to same actual source pose separates approximate pose from intrinsic command/curve shape. Low servo error does not imply successful contact; U03 is under load and its command-based predictions can deviate substantially.',cases=sets,selected=selected)
(dest/'material_shape_pose.json').write_text(json.dumps(out,indent=2));(dest/'raw_front_wave_fit.json').write_text(json.dumps(wave,indent=2))
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
for name,rows in sets.items():
    tt=[r['t'] for r in rows];ax[0].plot(tt,[r['actual_M0_to_M6_centers_m'][0][1] for r in rows],label=name+' actual M0 Y');ax[1].plot(tt,[sum(r['actual_M0_to_M6_target_N'][:6]) for r in rows],label=name+' front force');ax[2].plot(tt,[r['actual_M0_to_M6_polar_deg'][0] for r in rows],label=name+' actual M0 polar')
rows=sets['A15'];tt=[r['t'] for r in rows];ax[0].plot(tt,[r['nominal_M0_to_M6_centers_m'][0][1] for r in rows],'--',label='A15 C');ax[0].plot(tt,[r['shape_anchored_in_actual_source']['ctrl']['M0_to_M6_centers_m'][0][1] for r in rows],':',label='A15 ctrl FK in actual source pose');ax[2].plot(tt,[r['nominal_M0_to_M6_polar_deg'][0] for r in rows],'--',label='A15 C M0 polar')
for a in ax:a.legend();a.grid(alpha=.25);a.set_xlim(50,56.4);a.axvspan(53,54.9,color='gray',alpha=.12)
ax[0].set_ylabel('head center Y (m)');ax[1].set_ylabel('front6 target normal (N)');ax[2].set(ylabel='M0 target-axis polar (deg)',xlabel='saved actual time (s)');fig.suptitle('A16 diagnosis: actual bearing gap and nominal/native/world timing');fig.tight_layout();fig.savefig(dest/'handover_timing.png',dpi=150)
for name,rows in selected.items():
    print(name)
    for r in rows:
        h=r['actual_M0_to_M6_centers_m'][0];nh=r['shape_anchored_in_actual_source']['ctrl']['M0_to_M6_centers_m'][0];print(dict(t=r['t'],Y_actual=h[1],Y_native_actual_source=nh[1],Y_C=r['nominal_M0_to_M6_centers_m'][0][1] if r['nominal_M0_to_M6_centers_m'] else None,polar_actual=r['actual_M0_to_M6_polar_deg'][0],polar_C=r['nominal_M0_to_M6_polar_deg'][0] if r['nominal_M0_to_M6_polar_deg'] else None,front_target_N=r['actual_M0_to_M6_target_N'],prediction_error_M0_mm=r['shape_anchored_in_actual_source']['ctrl']['front_actual_prediction_error_mm'][0],tracking=r['tracking_max_deg']))
