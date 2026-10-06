"""Read saved round2 dynamics. No physical stepping or control dispatch."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import csv
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from curve_local_path import LocalPathFit
from snake_control.local_material_reference import BoundaryMaterialCurve
from snake_control.target_capture_reference import TargetCaptureReference

PARENT=ROOT/'experiments/curve_guided_transfer_20261006'
OLD=PARENT/'stage_continuation_20261006'
BASE=PARENT/'target_wrap_round2'


def analyze(case):
    z=np.load(case/'trace.npz');controls=json.loads((case/'control.json').read_text(encoding='utf8'))
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    t=z['sim_time'];ct=np.array([r['time_s'] for r in controls])
    force=z['contact_force_N'];p=z['points_mm'];center=z['centers_mm']
    speed=np.linalg.norm(np.diff(center,axis=0)/np.diff(t)[:,None,None],axis=2)
    polar=np.unwrap(np.arctan2(p[:,:9,2]-1000,p[:,:9,0]-1100),axis=1)
    turns=np.degrees(abs(polar[:,-1]-polar[:,0]));spans=np.degrees(np.ptp(polar,axis=1))
    fit=LocalPathFit()
    ref=TargetCaptureReference(case/'reference.npz') if case.name.startswith('CAPTURE') else BoundaryMaterialCurve(case/'reference.npz')
    times=np.unique(np.r_[np.arange(0,t[-1]+1e-8,.5),5,31.9,34.9,t[-1]])
    times=times[times<=t[-1]+1e-8];rows=[]
    for moment in times:
        i=int(abs(t-moment).argmin())
        # Physical samples at a command boundary precede the next dispatch.
        # Match the command actually applied to that saved frame, not the next
        # calculation sharing its nominal timestamp.
        ci=min(max(int(z['command_id'][i])-1,0),len(controls)-1);c=controls[ci]
        fit.d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(fit.m,fit.d);fit.last=None
        R=fit.d.body('head').xmat.reshape(3,3).copy();gaps=[float(fit.radial_gap(m,1)) for m in [0,6,7,8,9]]
        axis_errors=[];estimated_errors=[]
        for a,world in zip(c['material_axes'],[np.array([1.,0,0]),np.array([0.,1,0])]):
            truth=R.T@world
            axis_errors.append(float(np.degrees(np.arccos(np.clip(np.array(a['direction_head'])@truth,-1,1)))))
            estimated_errors.append(float(np.degrees(np.arccos(np.clip(np.array(a.get('estimated_direction_head',a['direction_head']))@truth,-1,1)))))
        fit.set(ref.state(float(c['feed_mm'])));nominal_gaps=[float(fit.radial_gap(m,1)) for m in [0,6,7,8,9]]
        rows.append(dict(time_s=float(t[i]),applied_command_id=int(z['command_id'][i]),control_time_s=float(c['time_s']),
            initial_frame_control_is_next_planned=bool(i==0),stage=str(z['stage'][i]),feed_mm=float(z['feed_mm'][i]),
            original_front_target_force_N=float(force[i,1,:8].sum()),original_bearing_M0_M6_M7_target_force_N=float(force[i,1,[0,6,7]].sum()),
            M8_target_force_N=float(force[i,1,8]),M9_target_force_N=float(force[i,1,9]),
            source_contact_modules=np.flatnonzero(force[i,0]>.1).tolist(),target_contact_modules=np.flatnonzero(force[i,1]>.1).tolist(),
            front_polar_turn_deg=float(turns[i]),front_polar_span_deg=float(spans[i]),
            projected_target_shell_gaps_M0_M6_M7_M8_M9_mm=gaps,nominal_shell_gaps_M0_M6_M7_M8_M9_mm=nominal_gaps,
            used_axis_error_deg=axis_errors,estimated_axis_error_deg=estimated_errors,
            J7_J10_weights=c['grasp_weights'][6:10],J7_J10_retained_deg=np.degrees(c['retained_shape_rad'][6:10]).tolist(),
            J7_J10_AD_deg=np.degrees(c['compliance_delta_rad'][6:10]).tolist(),
            J7_J10_receiving_AD_deg=np.degrees(c.get('receiving_delta_rad',np.zeros(20))[6:10]).tolist(),
            capture_phase=c.get('capture_phase',0),rate_projected_joints=c['rate_limited_joints'],
            saturated_joints_1based=(np.flatnonzero(abs(z['torque'][i])>=2.355-1e-6)+1).tolist(),
            maximum_module_speed_mm_s=float(speed[max(0,i-1)].max()) if i else 0))
    selected=[min(rows,key=lambda r:abs(r['time_s']-x)) for x in [0,5,34.9,50,56.5,61.8,float(t[-1])] if x<=t[-1]+1e-8]
    counts=np.count_nonzero(force>.1,axis=2)
    summary=dict(case=case.name,physics_steps_added=0,classification=result['classification'],formal_success_eligible=result['formal_success_eligible'],
        actual_duration_s=float(t[-1]),front_final_polar_turn_deg=float(turns[-1]),front_final_polar_span_deg=float(spans[-1]),
        front_final_target_force_N=float(force[-1,1,:8].sum()),
        M8_maximum_target_force_N=float(force[:,1,8].max()),M9_maximum_target_force_N=float(force[:,1,9].max()),
        M8_final_target_force_N=float(force[-1,1,8]),M9_final_target_force_N=float(force[-1,1,9]),
        two_pipe_minimum_module_contacts_after5=counts[t>=5-1e-8].min(axis=0).tolist() if t[-1]>=5 else None,
        maximum_module_speed_mm_s=float(speed.max()),maximum_torque_Nm=float(abs(z['torque']).max()),
        ground_frames=int(np.count_nonzero(z['ground'])),nonadjacent_self_contact_frames=int(np.count_nonzero(z['self_contact'])),
        external_applied_force_maximum=float(z['external_applied_force_maximum'].max()),generalized_applied_force_maximum=float(z['generalized_applied_force_maximum'].max()),
        original_front_target_contacts_final=np.flatnonzero(force[-1,1,:8]>.1).tolist(),
        selected_actual_samples=selected,original_result=result,
        descriptive_limits='Open-chain polar range is not enclosure topology; projected gaps require axial overlap; truth axis errors are diagnosis only.')
    (case/'round2_diagnostic_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (case/'round2_timeline.json').write_text(json.dumps(dict(physics_steps=0,rows=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,ax=plt.subplots(3,2,figsize=(11,9),sharex=True)
    for m in [0,6,7,8,9]:ax[0,0].plot(t,force[:,1,m],label='M'+str(m))
    ax[0,0].legend(ncol=3);ax[0,0].set_ylabel('Actual target normal force (N)')
    ax[0,1].plot(t,turns,label='Net polar turn');ax[0,1].plot(t,spans,label='Polar span');ax[0,1].legend();ax[0,1].set_ylabel('Original front angle (deg)')
    for j in [6,7,8,9]:ax[1,0].plot(ct,np.degrees([r['retained_shape_rad'][j] for r in controls]),label='J'+str(j+1))
    ax[1,0].legend();ax[1,0].set_ylabel('ACK retained shape (deg)')
    for j in [6,7,8,9]:ax[1,1].plot(ct,[r['grasp_weights'][j] for r in controls],label='J'+str(j+1))
    ax[1,1].legend();ax[1,1].set_ylabel('Compliance participation')
    tt=[r['time_s'] for r in rows]
    for side in [0,1]:ax[2,0].plot(tt,[r['used_axis_error_deg'][side] for r in rows],label=['Source','Target'][side])
    ax[2,0].legend();ax[2,0].set_ylabel('Used axis error (offline, deg)')
    ax[2,1].plot(t[1:],speed.max(axis=1));ax[2,1].set_ylabel('Maximum module speed (mm/s)')
    for a in ax.flat:a.grid(alpha=.2);a.axvspan(31.9,34.9,color='.8',alpha=.4);a.axvspan(61.8,66.8,color='r',alpha=.1)
    ax[2,0].set_xlabel('Physical time (s)');ax[2,1].set_xlabel('Physical time (s)')
    fig.suptitle(case.name+' / actual saved physics');fig.tight_layout();fig.savefig(case/'round2_actual_diagnostics.png',dpi=130);plt.close(fig)
    return summary


def main():
    summaries=[]
    for case in sorted(BASE.iterdir()):
        if case.is_dir() and (case/'result.json').exists() and (case/'trace.npz').exists():
            summaries.append(analyze(case))
    output=dict(physics_steps=0,cases=summaries)
    (BASE/'round2_diagnostic_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps([dict(case=s['case'],duration=s['actual_duration_s'],front_turn=s['front_final_polar_turn_deg'],front_force=s['front_final_target_force_N'],M9max=s['M9_maximum_target_force_N']) for s in summaries],indent=2),flush=True)


if __name__=='__main__':main()
