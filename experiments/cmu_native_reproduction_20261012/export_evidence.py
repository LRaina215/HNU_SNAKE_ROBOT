from native_run import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for name in ['N09','N10']:
    case=BASE/name;z=np.load(case/'trace.npz');t=z['t'];c=z['centers'];f=z['forces'];verified=json.loads((case/'verified_success.json').read_text())
    with (case/'material_events.csv').open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.DictWriter(out,fieldnames=list(verified['events'][0]));w.writeheader();w.writerows(verified['events'])
    rr=np.linalg.norm(c[:,:,[0,2]]-[1.1,1],axis=2)
    with (case/'timeline.csv').open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.writer(out);w.writerow(['time_s','phase_rad','nominal_nt','loosening','source_force_N','target_force_N','head_x_m','head_y_m','head_z_m','M18_x_m','M18_y_m','M18_z_m','max_joint_tracking_deg','max_pipe_depth_mm','self_depth_mm','all_centers_target_band'])
        for i in range(len(t)):w.writerow([t[i],z['phase'][i],z['nt'][i],z['L'][i],*f[i].sum(1),*z['points'][i,0],*c[i,18],np.rad2deg(z['tracking'][i]),max(z['depth'][i,:2])*1000,z['depth'][i,2]*1000,bool(np.all((rr[i]>.07)&(rr[i]<.12)&(c[i,:,1]>.15)&(c[i,:,1]<1.2)))])
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    axs[0].plot(t,f[:,0].sum(1),label='source');axs[0].plot(t,f[:,1].sum(1),label='target');axs[0].set_ylabel('Contact normal force / N');axs[0].legend()
    for n in [0,5,10,15,20]:axs[1].plot(t,rr[:,n]*1000,label=f'M{n}')
    axs[1].axhspan(70,120,color='green',alpha=.12);axs[1].set_ylabel('Actual distance to target axis / mm');axs[1].legend(ncol=5)
    axs[2].plot(t,z['nt'],color='purple',label='nominal nt');axs[2].set_ylabel('Nominal head-window boundary');ax=axs[2].twinx();ax.plot(t,c[:,18,0],label='M18 actual X',color='orange');ax.plot(t,c[:,18,1],label='M18 actual Y',color='green');ax.set_ylabel('Actual same material module position / m');ax.legend(loc='lower right');axs[2].legend(loc='upper left');axs[2].set_xlabel('Actual simulation time / s')
    for axis in axs:
        for tt in [verified['front_first_stable_time'],verified['last_source_contact_time'],verified['continuous_final_full_target_from']]:axis.axvline(tt,color='black',alpha=.3,ls='--')
        axis.grid(alpha=.2)
    fig.suptitle(f'{name}: actual contacts / target wrap / nominal window vs material motion');fig.tight_layout();fig.savefig(case/'evidence_timeline.png',dpi=160);plt.close(fig)
    print(name,'exports complete')
