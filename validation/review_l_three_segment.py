"""Read saved L-trial evidence; never run dynamics or alter controller files."""
from pathlib import Path
import sys,json,shutil,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.suspended_supply_transfer import nodes
from validation.suspended_supply_experiment import write
BASE=ROOT/'experiments/unified_curve_transfer_20261007/L_THREE_SEGMENT_20261009'

def review(case):
    frozen=case/'sources/mujoco_model/Snake_Robot_o20_L_three_segment.xml'
    for mesh in ET.parse(frozen).findall('./asset/mesh'):
        name=mesh.get('file')
        if name:
            dest=frozen.parent/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(ROOT/'mujoco_model'/name,dest)
    m=mujoco.MjModel.from_xml_path(str(frozen));d=mujoco.MjData(m)
    z=np.load(case/'trace.npz');t=z['sim_time'];c=z['centers_mm']/1000
    rr=np.linalg.norm(c[:,:,[0,2]]-[1.1,1.],axis=2)
    raw=np.arctan2(c[:,:,2]-1,c[:,:,0]-1.1);phi=np.unwrap(raw,axis=1)
    da=np.diff(phi,axis=1);dy=np.diff(c[:,:,1],axis=1)
    near=(rr>.065)&(rr<.125)&(c[:,:,1]>.15)
    conn=near.copy();conn[:,1:] &= near[:,:-1]&(da>.30)&(da<1.10)&(dy<-.002)&(dy>-.045)&(np.abs(dy+.180/(2*np.pi)*da)<.025)
    source=z['source_contact_any'].any(1);forces=z['contact_force_N'].sum(2)
    tail_force=z['contact_force_N'][:,1,18:].sum(1)
    ks=np.flatnonzero(t>=t[-1]-1.00001);error=0.;last=[]
    for i in ks:
        d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d)
        error=max(error,float(np.abs(nodes(d)*1000-z['points_mm'][i]).max()))
        fit=np.polyfit(phi[i],c[i,:,1],1);pitch=-fit[0]*2*np.pi
        rms=np.sqrt(np.mean((c[i,:,1]-np.polyval(fit,phi[i]))**2))
        strict_shape=bool(np.all((rr[i]>.075)&(rr[i]<.105)) and np.max(np.abs(rr[i]-np.median(rr[i])))<.006 and np.max(np.abs(dy[i]+.180/(2*np.pi)*da[i]))<.008 and .162<pitch<.198 and rms<.004)
        last.append(dict(t_s=float(t[i]),all_near=bool(near[i].all()),all_connected=bool(conn[i].all()),strict_actual_whole_shape=strict_shape,
            pitch_mm=float(pitch*1000),axis_rms_mm=float(rms*1000),source_contact=bool(source[i]),
            source_force_n=float(forces[i,0]),target_posterior_force_n=float(tail_force[i]),
            tail_dy_mm=float(dy[i,-1]*1000),tail_dphi_rad=float(da[i,-1]),all_full_kp=bool((z['kp_nm_per_rad'][i]>120).all()),all_kp_exact127=bool((z['kp_nm_per_rad'][i]>=126.999).all())))
    ids=[0,len(t)//2,len(t)-1];samples=[]
    for i in ids:
        samples.append(dict(t_s=float(t[i]),shift_mm=float(z['shift_mm'][i]),actual_points_mm=z['points_mm'][i],
            executed_reference_points_mm=z['reference_points_mm'][i],current_target_connected=np.flatnonzero(conn[i]),
            source_contacts=np.flatnonzero(z['source_contact_any'][i]),source_target_force_n=forces[i],kp_nm_per_rad=z['kp_nm_per_rad'][i]))
    connected_count=conn.sum(1)
    clear=(~source)&(forces[:,0]<.1);edges=np.diff(np.r_[False,clear,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
    intervals=[dict(start_s=float(t[a]),end_s=float(t[b]),duration_s=float(t[b]-t[a])) for a,b in zip(starts,ends) if t[b]-t[a]>=.5]
    joints=[]
    for j in range(20):
        joints.append(dict(joint=j+1,actual_angle_min_deg=float(np.degrees(z['q'][:,j].min())),actual_angle_max_deg=float(np.degrees(z['q'][:,j].max())),
            max_drive_torque_nm=float(abs(z['torque'][:,j]).max()),max_passive_torque_nm=float(abs(z['passive_torque_nm'][:,j]).max()),
            min_kp=float(z['kp_nm_per_rad'][:,j].min()),max_kp=float(z['kp_nm_per_rad'][:,j].max()),final_kp=float(z['kp_nm_per_rad'][-1,j]),
            original_damping=float(m.dof_damping[m.jnt_dofadr[m.actuator_trnid[j,0]]]),
            measured_supply_seconds=float(np.count_nonzero(z['kp_nm_per_rad'][:,j]<.5)*.01)))
    write(case/'joint_region_dynamics_summary.json',dict(physical_steps=0,joints=joints,source_clear_intervals_at_least_half_second=intervals,
        model=dict(gravity=m.opt.gravity,hinge_range_deg=np.degrees(m.jnt_range[1:]),force_range_nm=m.actuator_forcerange,
            mass_total_kg=float(m.body_mass[m.body('head').id:m.body('head').id+21].sum()),original_robot=True)))
    result=dict(case=case.name,read_only=True,physical_steps=0,samples_100hz=len(t),actual_duration_s=float(t[-1]),
        maximum_current_target_connected=int(connected_count.max()),maximum_current_target_connected_time_s=float(t[connected_count.argmax()]),
        final_actual_target_connected=np.flatnonzero(conn[-1]),final_source_contact_modules=np.flatnonzero(z['source_contact_any'][-1]),
        whole_source_clear_first_time_s=float(t[np.flatnonzero((~source)&(forces[:,0]<.1))[0]]) if ((~source)&(forces[:,0]<.1)).any() else None,
        source_clear_intervals_at_least_half_second=intervals,
        actual_pipe_depth_max_mm=float(z['pipe_depth_mm'].max()),actual_joint_max_deg=float(np.degrees(abs(z['q']).max())),
        last_second_fk_max_point_difference_mm=error,last_second_samples=len(ks),last_second_original_helix_guard=all(x['all_near'] and x['all_connected'] and 126<x['pitch_mm']<234 and x['axis_rms_mm']<12 for x in last),
        last_second_all_source_clear=all(not x['source_contact'] and x['source_force_n']<.1 for x in last),
        last_second_all_full_kp=all(x['all_full_kp'] for x in last),last_second_all_kp_exact127=all(x['all_kp_exact127'] for x in last),
        last_second_all_strict_whole_shape=all(x['strict_actual_whole_shape'] for x in last),last_second_all_full_reference_window=bool((z['shift_mm'][ks]>=838.5-1e-6).all()),
        last_second_tail_supported=all(x['target_posterior_force_n']>.2 for x in last),
        initial_mid_final=samples,last_second=last,yamada_compliance_enabled=False)
    kpmin=z['kp_nm_per_rad'].min(1)
    def final_interval(mask):
        bad=np.flatnonzero(~mask);start=int(bad[-1]+1) if len(bad) else 0
        return float(t[-1]-t[start]) if start<len(t) else 0.
    median_r=np.median(rr[ks,:20],axis=1);tail_radial_error=rr[ks,20]-median_r
    result['tail_and_completion_caveat']=dict(
        recorded_guard_used_radius_band_mm=[75,105] if case.name=='L04' else [65,125],recorded_guard_is_not_strict_uniform_radius_verification=case.name!='L04',
        final_all_center_radius_mm=rr[-1]*1000,final_first20_median_radius_mm=float(np.median(rr[-1,:20])*1000),
        last_second_tail_radius_range_mm=[float(rr[ks,20].min()*1000),float(rr[ks,20].max()*1000)],
        last_second_tail_radial_excess_over_first20_median_range_mm=[float(tail_radial_error.min()*1000),float(tail_radial_error.max()*1000)],
        final_tail_direct_force_n=float(z['contact_force_N'][-1,1,20]),last_second_tail_direct_force_max_n=float(z['contact_force_N'][ks,1,20].max()),
        last_second_tail_neighbor19_20_force_min_n=float(z['contact_force_N'][ks,1,19:].sum(1).min()),
        full_kp_above120_recorded_interval_s=final_interval(kpmin>120),full_kp127_recorded_interval_s=final_interval(kpmin>=126.999),
        final_reference_shift_mm=float(z['shift_mm'][-1]),nominal_final_reference_shift_mm=838.5,
        review_tail_radial_consistency_tolerance_mm=6.,
        strict_uniform_radius_tail_helix_verified=bool(np.max(np.abs(tail_radial_error))<=.006))
    write(case/'actual_evidence_review.json',result)
    fig=plt.figure(figsize=(16,9))
    for col,i in enumerate(ids):
        ax=fig.add_subplot(2,3,col+1,projection='3d');p=z['points_mm'][i]/1000;g=z['reference_points_mm'][i]/1000
        ax.plot([-.2,1.1],[0,0],[1,1],color='grey',lw=10,alpha=.25);ax.plot([1.1,1.1],[0,1.65],[1,1],color='grey',lw=10,alpha=.25)
        ax.plot(g[:,0],g[:,1],g[:,2],'--',lw=1.5,label='Executed reference');ax.plot(p[:,0],p[:,1],p[:,2],'o-',ms=3,label='Actual FK')
        ax.scatter(*p[-1],color='red');ax.set(xlim=(.45,1.40),ylim=(-.20,.90),zlim=(.4,1.2),xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)',title=f'{case.name} {t[i]:.2f}s / shift {z["shift_mm"][i]:.1f}mm')
        ax.view_init(27,140);ax.legend(fontsize=8)
    ax=fig.add_subplot(2,3,4);ax.plot(t,connected_count,label='Actual target connections');ax.plot(t,z['actual_exit_mask'][:,15:].sum(1),label='Registered source clear');ax.set(xlabel='Actual time(s)',ylabel='Rods');ax.legend()
    ax=fig.add_subplot(2,3,5);ax.plot(t,forces[:,0],label='Source');ax.plot(t,forces[:,1],label='Target');ax.set(xlabel='Actual time(s)',ylabel='Bearing normal force(N)');ax.legend()
    ax=fig.add_subplot(2,3,6);ax.plot(t,z['actual_front_span_rad'],label='Front wrap(rad)');ax.plot(t,z['pipe_depth_mm'],label='Actual penetration(mm)');ax.set(xlabel='Actual time(s)');ax.legend()
    fig.tight_layout();fig.savefig(case/'actual_and_executed_reference.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(16,5))
    for col,i in enumerate(ids):
        ax=fig.add_subplot(1,3,col+1,projection='3d');g=z['reference_points_mm'][i]/1000
        ax.plot([-.2,1.1],[0,0],[1,1],color='grey',lw=10,alpha=.25);ax.plot([1.1,1.1],[0,1.65],[1,1],color='grey',lw=10,alpha=.25)
        ax.plot(g[:,0],g[:,1],g[:,2],'o-',ms=3);ax.scatter(*g[-1],color='red');ax.set(xlim=(.45,1.40),ylim=(-.20,.90),zlim=(.4,1.2),title=f'Executed reference at {t[i]:.2f}s');ax.view_init(27,140)
    fig.tight_layout();fig.savefig(case/'executed_reference_initial_middle_final.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['initial_mid_final','last_second']},default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':
    for name in sys.argv[1:] or ['L01','L02','L03','L04']:review(BASE/name)
