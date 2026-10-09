"""Read-only current-window failure review and evidence for R01-R03."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,shutil,importlib.util
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.ready_inner_entry_experiment import BASE
from validation.suspended_supply_experiment import write
from snake_control.suspended_supply_transfer import nodes

def review(name):
    case=BASE/name;z=np.load(case/'trace.npz');t=z['sim_time'];p=z['points_mm']/1000;c=z['centers_mm']/1000;f=z['contact_force_N'];commands=json.loads((case/'commands.json').read_text());rej=json.loads((case/'rejected_current_command.json').read_text());result=json.loads((case/'result.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'));d=mujoco.MjData(m);fk=0.
    ids=sorted(set([0,len(t)//2,len(t)-1]));stages=[]
    for i in ids:
        d.qpos[:]=z['qpos'][i];mujoco.mj_fwdPosition(m,d);fk=max(fk,float(np.linalg.norm(nodes(d)-p[i],axis=1).max()*1000));stages.append(dict(t_s=float(t[i]),actual_points_mm=p[i]*1000,reference_points_mm=z['reference_points_mm'][i],source_target_force_n=f[i].sum(1)))
    projection=rej['projection'];d.qpos[:]=z['qpos'][-1]
    if 'registered_math_root' in projection:d.qpos[:7]=projection['registered_math_root'];d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]=projection['geometry_q_rad']
    else:
        spec=importlib.util.spec_from_file_location('frozen_ready_'+name,case/'sources/snake_control/head_first_inner_route.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);route=mod.HeadFirstInnerRoute(p[0],smooth_source=False)
        initial_R=Rotation.from_quat(z['qpos'][0,3:7][[1,2,3,0]]).as_matrix();rot=route.head_orientation(initial_R,rej['local_feed_shift_mm']/1000,rej['roll_phase_rad'])@Rotation.from_rotvec(projection['private_head_tilt_rad']).as_matrix()
        d.qpos[:3]=projection['private_root'];d.qpos[3:7]=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]];d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]=rej['ctrl_rad']
    mujoco.mj_fwdPosition(m,d);candidate=nodes(d);contacts=[]
    for con in d.contact:
        names=[m.geom(con.geom1).name,m.geom(con.geom2).name]
        if any((x or '').startswith('pipe_') for x in names):
            robot=con.geom2 if (names[0] or '').startswith('pipe_') else con.geom1
            contacts.append(dict(geometry_names=names,robot_module=int(m.geom_bodyid[robot]-m.body('head').id),depth_mm=max(0.,-float(con.dist))*1000))
    sr=np.linalg.norm(c[0][:,[1,2]]-[0,1],axis=1)
    diff=(p[-1,0]-p[0,0])*1000;headerr=np.linalg.norm(p[:,0]*1000-z['reference_points_mm'][:,0],axis=1)
    audit=dict(name=name,physical_steps_in_review=0,outcome=result['outcome'],actual_seconds=float(t[-1]),actual_head_displacement_mm=diff,actual_head_displacement_norm_mm=float(np.linalg.norm(diff)),maximum_world_head_error_mm=float(headerr.max()),initial_source_radius_mm=sr*1000,initial_target_force_n=float(f[0,1].sum()),maximum_target_force_n=float(f[:,1].sum(1).max()),final_source_target_force_n=f[-1].sum(1),maximum_actual_pipe_depth_mm=float(z['pipe_depth_mm'].max()),maximum_actual_self_depth_mm=float(z['self_depth_mm'].max()),maximum_actual_joint_deg=float(np.degrees(abs(z['q']).max())),frozen_fk_node_error_mm=fk,actual_dispatched_commands=len(commands),explicit_anchor_roll_target_commands=sum(x['projection'].get('explicit_roll_target_applied',False) for x in commands),current_rejected_projected_pipe_depth_mm=projection['predicted_pipe_depth_mm'],current_rejected_fixed_root_pipe_depth_mm=rej['current_command_pipe_depth_mm'],current_rejected_fit_rms_mm=projection.get('whole_fit_rms_mm',projection.get('curve_fit_rms_mm')),reconstructed_current_candidate_pipe_contacts=sorted(contacts,key=lambda x:x['depth_mm'],reverse=True),whole_transfer_complete=False,active_target_acquisition=False,physical_root_reset=False,yamada_compliance_enabled=False)
    audit['last_dispatched_shift_mm']=commands[-1]['shift_mm'];audit['next_rejected_shift_mm']=rej['shift_mm'];audit['reference_head_displacement_at_actual_stop_mm']=z['reference_points_mm'][-1,0]-z['points_mm'][0,0]
    audit['maximum_issued_reference_rod_error_mm']=max(float(abs(np.linalg.norm(np.diff(np.asarray(x['reference_points_mm']),axis=0),axis=1)-64.5).max()) for x in commands)
    audit['next_rejected_joint_target_change_deg']=rej['maximum_dispatch_step_deg']
    write(case/'actual_review.json',audit);write(case/'stage_comparison.json',stages);write(case/'rejected_candidate_pose.json',dict(status='Private predicted pose, not physically dispatched',points_mm=candidate*1000,contacts=contacts))
    fig,axes=plt.subplots(2,4,figsize=(17,8));ii=[0,len(t)//2,len(t)-1,None]
    for k,i in enumerate(ii):
        pp=candidate if i is None else p[i];ref=np.asarray(rej['reference_points_mm'])/1000 if i is None else z['reference_points_mm'][i]/1000
        for row,(a,b) in enumerate([(0,1),(0,2)]):
            ax=axes[row,k];ax.plot(pp[:,a],pp[:,b],'o-',ms=2.5,label='Rejected private prediction' if i is None else 'Actual body');ax.plot(ref[:,a],ref[:,b],'--',label='Current world guide');ax.scatter(pp[0,a],pp[0,b],c='red',label='Head')
            if row==0:ax.plot([-.2,1.1],[0,0],color='gray',lw=9,alpha=.25);ax.plot([1.1,1.1],[0,2],color='gray',lw=9,alpha=.25);ax.set(xlim=(.4,1.25),ylim=(-.2,.5))
            else:ax.axhline(1,color='gray',ls=':');th=np.linspace(0,2*np.pi,100);ax.fill(1.1+.0565*np.cos(th),1+.0565*np.sin(th),color='gray',alpha=.25);ax.set(xlim=(.4,1.25),ylim=(.74,1.2))
            ax.set(title='Next command rejected; not executed' if i is None else f'Actual t={t[i]:.2f}s',xlabel='World X(m)',ylabel='World Y(m)' if row==0 else 'World Z UP(m)',aspect='equal');ax.legend(fontsize=6)
    fig.suptitle(name+': no target contact; current command stopped before dispatch');fig.tight_layout();fig.savefig(case/'actual_and_rejected_stages.png',dpi=145);plt.close(fig)
    curve=json.loads((case/'complete_curve.json').read_text());v=np.array(curve['stations_mm']);cp=np.array(curve['points_mm'])/1000;join=curve['target_join_mm'];fig=plt.figure(figsize=(13,7));ax=fig.add_subplot(121,projection='3d')
    for mask,label,color in [(v<0,'Source','darkorange'),((v>=0)&(v<join),'Bridge','purple'),(v>=join,'Target planned','darkgreen')]:ax.plot(*cp[mask].T,label=label,color=color)
    ax.plot(*p[0].T,'o-',ms=2,label='Artificial all-source start');ax.plot(*p[-1].T,'o-',ms=2,label='Actual stop');ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)');ax.view_init(27,140);ax.legend(fontsize=7)
    ax=fig.add_subplot(122);ax.plot(t,p[:,0,1]*1000,label='Actual head Y');ax.plot(t,(p[:,0,2]-1)*1000,label='Actual head Z offset');ax.plot(t,headerr,label='Actual head world guide error');ax.plot(t,f[:,0].sum(1),label='Source bearing N');ax.plot(t,f[:,1].sum(1),label='Target bearing N');ax.set(xlabel='Actual time(s)',ylabel='mm / N');ax.legend();fig.tight_layout();fig.savefig(case/'complete_route_and_actual_progress.png',dpi=150);plt.close(fig)
    early=(v>=0)&(v<=200.2);front=cp[early]*1000;stations=v[early];actual=p[:,0]*1000;ref=z['reference_points_mm'][:,0];s=z['shift_mm']
    fig,axes=plt.subplots(1,3,figsize=(17,6))
    for ax,(a,b) in zip(axes[:2],[(0,1),(0,2)]):
        ax.plot(front[:,a],front[:,b],color='gray',lw=2,label='Planned first200mm (unexecuted part included)');ax.plot(ref[:,a],ref[:,b],'--',color='darkorange',label='Head reference actually issued');ax.plot(actual[:,a],actual[:,b],color='steelblue',label='Actual head');ax.scatter(actual[0,a],actual[0,b],c='green',label='Artificial start');ax.scatter(actual[-1,a],actual[-1,b],c='red',label='Actual stop');ax.set(xlabel='World X(mm)',ylabel='World Y(mm)' if b==1 else 'World Z UP(mm)',aspect='equal');ax.legend(fontsize=7)
    ax=axes[2];ax.plot(stations,front[:,2]-p[0,0,2]*1000,color='gray',label='Planned head Z change');ax.plot(s,ref[:,2]-p[0,0,2]*1000,'--',color='darkorange',label='Issued reference Z change');ax.plot(s,actual[:,2]-p[0,0,2]*1000,color='steelblue',label='Actual Z change');ax.axvline(commands[-1]['shift_mm'],color='red',ls=':',label='Last physically issued window');ax.set(xlabel='Shared shift(mm); first200mm enlarged',ylabel='Z change from artificial start(mm)');ax.legend(fontsize=7);fig.suptitle(name+': planned bridge itself first rises, then dips before upward target entry');fig.tight_layout();fig.savefig(case/'first200mm_head_reference_and_actual.png',dpi=150);plt.close(fig)
    write(case/'first200mm_review.json',dict(planned_stations_mm=stations,planned_head_points_mm=front,last_issued_shift_mm=commands[-1]['shift_mm'],next_rejected_shift_mm=rej['shift_mm'],reference_head_delta_at_actual_stop_mm=ref[-1]-p[0,0]*1000,actual_head_delta_mm=actual[-1]-actual[0],planned_first200_z_range_mm=[float(front[:,2].min()),float(front[:,2].max())],interpretation='Reference bridge has a downward portion. Actual descent is not attributable solely to the controller; reference/world execution difference remains separately measured.'))
    dest=case/'review_sources/validation/review_ready_inner_entry.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'validation/review_ready_inner_entry.py',dest)
    print(name,'actual displacement',diff,'rejected contacts',sorted(contacts,key=lambda x:x['depth_mm'],reverse=True)[:2],flush=True)

if __name__=='__main__':
    for name in ['R01','R02','R03']:review(name)
