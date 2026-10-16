"""Review saved H01-H03 only. Forward geometry, never mj_step or controls."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.suspended_supply_transfer import nodes

BASE=ROOT/'experiments/unified_curve_transfer_20261007/SUSPENDED_SUPPLY_20261009'
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def review(name):
    case=BASE/name
    with np.load(case/'trace.npz') as archive:z={k:archive[k] for k in archive.files}
    commands=json.loads((case/'commands.json').read_text(encoding='utf-8'))
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model);head=model.body('head').id
    source={model.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']}
    age=np.zeros(6);counts=[];current=[];whole=[];first={}
    for k in range(len(z['sim_time'])):
        data.qpos[:]=z['qpos'][k];mujoco.mj_fwdPosition(model,data)
        c=z['centers_mm'][k]/1000;radius=np.linalg.norm(c[15:,[1,2]]-[0.,1.],axis=1)
        contact=np.zeros(21,dtype=bool)
        for con in data.contact:
            for pipe,robot in [(con.geom1,con.geom2),(con.geom2,con.geom1)]:
                body=int(model.geom_bodyid[robot])
                if pipe in source and head<=body<head+21 and con.dist<=0:contact[body-head]=True
        dt=float(z['sim_time'][k]-z['sim_time'][k-1]) if k else 0.
        age=np.where((radius>.130)&(~contact[15:]),age+dt,0.)
        cleared=age>=.5;ids=np.flatnonzero(cleared)+15
        for j in ids:
            if str(j) not in first:first[str(j)]=float(z['sim_time'][k])
        counts.append(int(cleared.sum()));current.append(ids.tolist());whole.append(not np.any(contact))
    accepted=[x for x in commands if x['candidate_accepted']];rejected=[x for x in commands if not x['candidate_accepted']]
    mismatch=max((float(np.max(np.abs(np.array(x['ctrl_rad'])-np.array(x['mapper']['geometry_q_rad'])))) for x in accepted),default=0.)
    qcmd=np.array([x['ctrl_rad'] for x in commands]);firststep=np.max(np.abs(qcmd[0]-z['ctrl'][0])) if len(qcmd) else 0.
    step=max(firststep,float(np.max(np.abs(np.diff(qcmd,axis=0)))) if len(qcmd)>1 else 0.)
    source_zero=np.flatnonzero(np.array(whole));final_c=z['centers_mm'][-1]/1000
    r=json.loads((case/'result.json').read_text(encoding='utf-8'))
    row=dict(name=name,outcome=r['outcome'],actual_seconds=r['actual_seconds'],initial_source_physical_contact_modules=np.flatnonzero(z['contact_force_N'][0,0]>.1).tolist(),
        audited_current_source_exit_modules=current[-1],audited_source_exit_length_mm=len(current[-1])*.0645*1000,
        actual_source_exit_first_times_s=first,first_all_six_current_clear_s=float(z['sim_time'][next(i for i,x in enumerate(counts) if x==6)]) if 6 in counts else None,
        first_whole_source_no_contact_s=float(z['sim_time'][source_zero[0]]) if len(source_zero) else None,whole_source_no_contact_at_end=whole[-1],
        actual_new_retained_modules=r['actually_retained_new_modules'],reference_shift_mm=r['reference_shift_mm'],final_shape_parameter=r['final_shape_parameter'],
        candidate_accepted_count=len(accepted),candidate_count=len(commands),accepted_candidate_dispatch_max_error_deg=float(np.degrees(mismatch)),
        maximum_dispatch_step_deg=float(np.degrees(step)),maximum_command_deg=float(np.degrees(abs(qcmd)).max()) if len(qcmd) else 0.,
        first_rejection_s=rejected[0]['t_s'] if rejected else None,first_rejection_predicted_pipe_mm=rejected[0]['mapper']['predicted_pipe_depth_mm'] if rejected else None,
        first_rejection_soft_allowance_mm=rejected[0].get('predicted_contact_allowance_mm',1.2) if rejected else None,
        maximum_actual_joint_deg=r['maximum_actual_joint_deg'],maximum_actual_torque_nm=r['maximum_torque_nm'],
        maximum_actual_pipe_depth_mm=r['maximum_pipe_depth_mm'],maximum_actual_self_depth_mm=r['maximum_self_depth_mm'],ground=r['ground'],
        initial_front_span_rad=r['front_wrap_initial_rad'],final_front_span_rad=r['front_wrap_final_rad'],
        initial_tail_endpoint_z_mm=float(z['points_mm'][0,-1,2]),final_tail_endpoint_z_mm=float(z['points_mm'][-1,-1,2]),
        mean_free_body_vertical_tangent_fraction=float(np.mean(np.abs(np.diff(z['points_mm'][-1,9:],axis=0)[:,2])/.0645/1000)),
        final_target_bearing_modules=np.flatnonzero(z['contact_force_N'][-1,1]>.1).tolist(),
        final_target_normal_n=float(z['contact_force_N'][-1,1].sum()),all_save_errors=r['save_errors'],
        limitation='source exits are current radial geometry >130mm and no original source mesh contact for0.5s; not proof of controlled 6->4->2->0 timing or feed',
        h01_score_correction='H01 original distance-query exit score is not used; original files and video panel retained' if name=='H01' else None)
    write(case/'saved_record_review.json',row)
    np.savez_compressed(case/'audited_exit_series.npz',time=z['sim_time'],current_exit_count=np.array(counts),whole_source_no_contact=np.array(whole))
    return row,z,np.array(counts),commands

def main():
    allrows=[review(name) for name in ['H01','H02','H03']]
    write(BASE/'comparison.json',dict(status='three_attempt_limit_reached_no_full_transfer',physics_attempts=3,new_physics_steps=0,runs=[x[0] for x in allrows]))
    fig,axs=plt.subplots(3,3,figsize=(14,9),sharex='col')
    for k,(row,z,count,commands) in enumerate(allrows):
        t=z['sim_time'];axs[k,0].plot(t,count,label='Current source rods clear')
        axs[k,0].plot(t,z['actual_receive_mask'][:,9:].sum(axis=1),label='New target rods retained')
        axs[k,0].set_ylim(-.2,6.5);axs[k,0].set_ylabel(row['name']+' / rods');axs[k,0].legend(fontsize=8)
        axs[k,1].plot(t,z['points_mm'][:,-1,2],label='Actual tail endpoint Z')
        axs[k,1].axhline(227.48,color='gray',ls='--',label='Nominal hanging endpoint')
        axs[k,1].set_ylabel('mm');axs[k,1].legend(fontsize=8)
        tc=np.array([x['t_s'] for x in commands]);pred=np.array([x['mapper']['predicted_pipe_depth_mm'] for x in commands])
        allow=np.array([x.get('predicted_contact_allowance_mm',1.2) for x in commands])
        axs[k,2].plot(tc,pred,label='Predicted candidate shell depth')
        axs[k,2].plot(tc,allow,ls='--',label='Acceptance budget')
        axs[k,2].plot(t,z['pipe_depth_mm'],label='Actual shell depth')
        axs[k,2].set_ylabel('mm');axs[k,2].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Actual simulation time / s')
    fig.suptitle('Saved actual dynamics: source exit occurred; hanging supply and haul not completed',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(BASE/'actual_evidence.png',dpi=150);plt.close(fig)
    print(json.dumps([dict(name=r[0]['name'],current_source_exit=r[0]['audited_current_source_exit_modules'],
        first_all_clear_s=r[0]['first_all_six_current_clear_s'],candidate_dispatch_error_deg=r[0]['accepted_candidate_dispatch_max_error_deg'],
        final_tail_z_mm=r[0]['final_tail_endpoint_z_mm']) for r in allrows]))

if __name__=='__main__':main()
