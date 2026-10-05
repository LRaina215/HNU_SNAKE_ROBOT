"""Preserve the bounded window experiments, including unsuccessful results.

Checks actual relay delivery against planned interpolation; no new physics or
hash checks. Extract each ZIP into OUT to replay its saved trajectory.
"""
from pathlib import Path
import sys

# 支持从 codes/ 或工具目录直接运行，公共模块位于仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from segmented_pole_experiment import write_json
from window_coordination_experiment import OUT
from three_segment_review import context


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def delivery(folder):
    plan=read(folder/'plan.json')['arguments']
    controls=read(folder/'controls.json')
    commands=read(folder/'commands.json')
    total_commands=len(commands);probe_commands=0
    assessment=read(folder/'assessment.json')
    assert assessment['initial_pose_exact'] and assessment['EXE_delivery_checked']
    count=round(plan['dt']/plan['delivery_period'])
    with np.load(folder/'trace.npz') as z:previous=z['ctrl'][0].copy()
    if plan.get('dynamic_observer',False):
        probe=read(folder/'response_probe.json');probe_plan=read(folder/'response_probe_plan.json')
        probe_commands=probe_plan['commands'];assert len(probe)==probe_commands+1
        _,probe_trace,probe_model,_,_=context(folder)
        probe_execution=commands
        if plan.get('response_state'):
            # Natural approach already executed the probe. Verify those actual
            # commands in the parent trace; do not count/replay them a second
            # time in the post-probe transfer stage.
            parent=folder.parent/'approach'
            parent_trace=np.load(parent/'trace.npz')
            for key in ('time','qpos','qvel','ctrl'):
                assert np.allclose(parent_trace[key][-1],probe_trace[key][0],atol=1e-10,rtol=0)
            probe_trace=parent_trace
            probe_execution=[c for c in read(parent/'commands.json')
                if probe[0]['execution_time']-1e-8<=c['time']<probe[-1]['execution_time']-1e-8]
            assert len(probe_execution)==probe_commands
        probe_times=probe_trace['time'];probe_angles=probe_trace['qpos'][:,probe_model.jnt_qposadr[probe_model.actuator_trnid[:,0]]]
        probe_bounds=probe_model.jnt_range[probe_model.actuator_trnid[:,0]]
        for i,record in enumerate(probe[:-1]):
            t=record['time'];pulse=np.sin(2*np.pi*np.arange(1,4)*(t-.2)/.8) if .2<=t<=1. else np.zeros(3)
            expected=np.clip(np.asarray(probe_plan['reference'])+pulse@np.asarray(probe_plan['patterns']),probe_bounds[:,0]+.01,probe_bounds[:,1]-.01)
            assert np.allclose(record['command'],expected,atol=1e-12,rtol=0)
            assert np.allclose(probe_execution[i]['target'],expected,atol=1e-12,rtol=0)
            assert abs(probe_execution[i]['time']-record['execution_time'])<1e-8
            sample=int(np.argmin(abs(probe_times-record['execution_time'])))
            assert np.allclose(record['q'],probe_angles[sample],atol=1e-8,rtol=0)
            assert abs(probe_execution[i]['phase']-probe_plan['phase'])<1e-12
        previous=np.asarray(probe_execution[probe_commands-1]['target'])
        if plan.get('response_state'):probe_commands=0
        else:commands=commands[probe_commands:]
    assert len(commands)==len(controls)*count
    if plan.get('velocity_feedback',0):
        _,trace,model,_,_=context(folder)
        joint_ids=model.actuator_trnid[:,0]
        qadr=model.jnt_qposadr[joint_ids]
        bounds=model.jnt_range[joint_ids]
        trace_time=trace['time'];trace_angles=trace['qpos'][:,qadr]
        first=int(np.argmin(abs(trace_time-commands[0]['time']))) if probe_commands else 0
        previous_encoder=trace_angles[first].copy()
        filtered_velocity=np.zeros_like(previous_encoder)
        if plan.get('body_damping',0):
            from snake_control.encoder_shape_kinematics import EncoderShapeKinematics
            reference_module=plan.get('body_reference_module',0)
            if reference_module in (-1,-2):
                reference_module=read(folder/'plan.json')['configuration']['initial_front']
            kinematics=EncoderShapeKinematics(model,reference_module)
    largest=0.;largest_outer=0.
    for i,control in enumerate(controls):
        if plan.get('body_reference_module')==-2 and plan.get('body_damping',0):
            initial=read(folder/'plan.json')['configuration']['initial_front']
            reference_module=initial
            while reference_module<len(control['target']) and control['acquired_weight'][reference_module]>.9:
                reference_module+=1
            reference_module=min(reference_module,max(initial,int(control['formation_progress'])))
            assert reference_module==control['body_reference_module']
            kinematics.reference_module=reference_module
        target=np.asarray(control['target'])
        outer_increment=float(np.degrees(abs(target-previous)).max())
        largest_outer=max(largest_outer,outer_increment)
        assert outer_increment<=plan.get('shape_speed',90.)*plan['dt']+1e-8
        actual=np.asarray([c['target'] for c in commands[i*count:(i+1)*count]])
        fractions=np.arange(1,count+1)/count
        nominal=previous+(target-previous)*fractions[:,None]
        if plan.get('velocity_feedback',0):
            feedback=control['delivery_feedback'];assert len(feedback)==count
            old=previous.copy()
            for j,row in enumerate(feedback):
                assert np.allclose(row['nominal'],nominal[j],atol=1e-12,rtol=0)
                assert np.allclose(row['nominal_velocity'],(target-previous)/plan['dt'],atol=1e-12,rtol=0)
                ci=i*count+j
                sample=int(np.argmin(abs(trace_time-commands[ci]['time'])))
                assert abs(trace_time[sample]-commands[ci]['time'])<1e-8
                assert np.allclose(row['measured'],trace_angles[sample],atol=1e-8,rtol=0)
                assert np.allclose(row['previous_measured'],previous_encoder,atol=1e-8,rtol=0)
                velocity=(np.asarray(row['measured'])-row['previous_measured'])/plan['delivery_period']
                weight=1-np.asarray(control['acquired_weight']) if plan.get('velocity_scope')=='unacquired' else np.ones_like(target)
                assert np.allclose(row.get('weight',np.ones_like(target)),weight,atol=1e-12,rtol=0)
                if 'previous_filtered_velocity' in row:
                    assert np.allclose(row['previous_filtered_velocity'],filtered_velocity,atol=1e-10,rtol=0)
                alpha=plan['delivery_period']/(plan['delivery_period']+plan.get('velocity_filter',0))
                filtered_velocity+=alpha*(velocity-filtered_velocity)
                if 'filtered_velocity' in row:
                    assert np.allclose(row['filtered_velocity'],filtered_velocity,atol=1e-10,rtol=0)
                correction=-plan['velocity_feedback']*weight*(filtered_velocity-np.asarray(row['nominal_velocity']))
                assert np.allclose(row['measured_velocity'],velocity,atol=1e-12,rtol=0)
                assert np.allclose(row['correction'],correction,atol=1e-12,rtol=0)
                assert np.allclose(actual[j],row['target'],atol=1e-12,rtol=0)
                limit=np.radians(90)*plan['delivery_period']
                reference=nominal[j].copy()
                if plan.get('delivery_torque_equivalent'):
                    error=nominal[j]-np.asarray(row['measured'])
                    torque_error=model.actuator_forcerange[:,1]/model.actuator_gainprm[:,0]
                    reference+=weight*(np.clip(error,-torque_error,torque_error)-error)
                assert np.allclose(row.get('reference',nominal[j]),reference,atol=1e-12,rtol=0)
                body_correction=np.zeros_like(target)
                if plan.get('body_damping',0):
                    jacobians=kinematics.jacobians(np.asarray(row['measured']))
                    velocities=jacobians@filtered_velocity
                    speeds=np.linalg.norm(velocities,axis=1);module=int(speeds.argmax())
                    assert module==row['body_module']
                    assert np.allclose(row['body_jacobian'],jacobians[module],atol=1e-10,rtol=0)
                    assert np.allclose(row['body_velocity'],velocities[module],atol=1e-10,rtol=0)
                    assert abs(row['body_relative_speed']-speeds[module])<1e-10
                    force=-plan['body_damping']*max(0,1-plan['body_speed_limit']/max(speeds[module],1e-12))*velocities[module]
                    torque=jacobians[module].T@force
                    if reference_module:
                        assert np.max(abs(jacobians[module,:,:reference_module]))<1e-10
                        assert np.max(abs(torque[:reference_module]))<1e-8
                    body_correction=torque/model.actuator_gainprm[:,0]
                    assert np.allclose(row['body_damping_torque'],torque,atol=1e-9,rtol=0)
                    assert np.allclose(row['body_correction'],body_correction,atol=1e-11,rtol=0)
                    assert abs(row['body_feedback_power']-torque@filtered_velocity)<1e-9
                    assert torque@filtered_velocity<=1e-10
                expected=old+np.clip(reference+correction+body_correction-old,-limit,limit)
                expected=np.clip(expected,bounds[:,0]+.01,bounds[:,1]-.01)
                assert np.allclose(actual[j],expected,atol=1e-12,rtol=0)
                old=actual[j]
                previous_encoder=np.asarray(row['measured'])
        else:
            assert np.allclose(actual,nominal,atol=1e-12,rtol=0)
        largest=max(largest,float(np.degrees(abs(np.diff(np.vstack([previous,actual]),axis=0))).max()))
        previous=actual[-1]
    assert largest<=90*plan['delivery_period']+1e-8
    return dict(command_count=total_commands,response_probe_commands=probe_commands,response_probe_checked=bool(plan.get('dynamic_observer')),
        probe_executed_in_natural_approach=bool(plan.get('response_state')),interpolation_matches=not bool(plan.get('velocity_feedback',0)),
        velocity_feedback_checked=bool(plan.get('velocity_feedback',0)),
        maximum_outer_reference_increment_deg=largest_outer,
        maximum_command_increment_deg=largest,initial_pose_and_EXE_delivery_checked=True)


def coordination_plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    controls=read(folder/'controls.json')
    t=np.array([c['time'] for c in controls])
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for key,label in [('formation_progress','Formation command'),('actual_prefix','Transfer prefix proxy'),('rear','Rear boundary')]:
        axes[0].plot(t,[c[key] for c in controls],label=label)
    axes[1].plot(t,[c['front_wrap'] for c in controls],label='Measured front angular wrap')
    axes[1].axhline(.5,color='grey',ls='--',lw=.8)
    axes[2].plot(t,[c['phase_difference'] for c in controls],label='Rear minus front phase')
    for key,label in [('feed','Rear supply'),('follow','Command following'),('grip','Wrap/contact proxy')]:
        axes[3].plot(t,[c[key] for c in controls],label=label)
    for ax,label in zip(axes,['Joint index / count','Turns','Radians','Factor']):
        ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Physical seconds after checkpoint')
    fig.suptitle(folder.name+'\nTransfer prefix starts at the captured front count; not a support certificate.')
    fig.tight_layout()
    out=folder.parent/'reviews';out.mkdir(exist_ok=True)
    fig.savefig(out/(folder.name+'_coordination.png'),dpi=130);plt.close(fig)


def diagnose(folder):
    result,z,m,d,meta=context(folder)
    t=z['time']-z['time'][0]
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    speed=np.r_[np.zeros((1,21)),np.linalg.norm(np.diff(centers,axis=0),axis=2)/np.diff(t)[:,None]]
    report=read(folder/'process_metrics.json')
    valid=(t>=1)&(t<=report['metrics_end_seconds']+1e-6)
    row,module=np.unravel_index(np.where(valid[:,None],speed,-1).argmax(),speed.shape)
    controls=read(folder/'controls.json')
    ci=max(0,np.searchsorted([c['time'] for c in controls],t[row],side='left'))
    control=controls[min(ci,len(controls)-1)]
    peak=dict(time_seconds=float(t[row]),module=int(module),speed_mm_s=float(speed[row,module]),
        source_contact_modules=np.flatnonzero(z['normal_force'][row,0]>.1).tolist(),
        target_contact_modules=np.flatnonzero(z['normal_force'][row,1]>.1).tolist(),
        target_region_modules=int(z['region'][row].sum()),ground=bool(z['ground'][row]),
        commanded_partition=z['partition'][row].tolist(),
        formation_progress=control['formation_progress'],front_wrap_proxy=control['front_wrap'])
    source=np.any(z['normal_force'][:,0]>.1,axis=1)
    absent=np.convolve((~source).astype(int),np.ones(60,dtype=int),mode='full')[:len(t)]>=60
    ids=np.flatnonzero(absent)
    out=dict(peak_module_speed=peak,
        first_source_absence_0_6s_end=float(t[ids[0]]) if len(ids) else None,
        assessment=read(folder/'assessment.json'),
        limitation='Angular wrap and contact presence do not establish force closure; prefix has a captured-front floor.')
    write_json(folder/'failure_diagnosis.json',out)
    return out


def package(root=OUT,names=None):
    destination=root/'archives';destination.mkdir(exist_ok=True)
    index={row['run']:row for row in read(root/'runs.json')} if names and (root/'runs.json').exists() else {}
    for folder in sorted(root.iterdir()):
        if names and folder.name not in names:continue
        if not (folder/'stage1_gate.json').is_file():continue
        check=delivery(folder);write_json(folder/'delivery_check.json',check)
        result=read(folder/'result.json');plan=read(folder/'plan.json')
        controls=read(folder/'controls.json')
        diagnosis=diagnose(folder)
        coordination_plot(folder)
        output=destination/(folder.name+'.zip')
        files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.png','.mp4','.pyc'))
        with ZipFile(output,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
            for path in files:archive.write(path,path.relative_to(root).as_posix())
        index[folder.name]=dict(run=folder.name,archive=output.relative_to(root).as_posix(),
            hypothesis=result['hypothesis'],settings=plan['arguments'],outcome=result['outcome'],
            metrics=read(folder/'process_metrics.json'),acceptance=read(folder/'whole_crossing_audit.json'),
            stage1=read(folder/'stage1_gate.json'),delivery=check,diagnosis=diagnosis,
            coordination=dict(maximum_phase_difference=float(max(abs(c['phase_difference']) for c in controls)),
                initial_front_wrap=controls[0]['front_wrap'],final_front_wrap=controls[-1]['front_wrap'],
                maximum_formation=max(c['formation_progress'] for c in controls)),
            files=len(files),bytes=output.stat().st_size)
        print(folder.name,round(output.stat().st_size/1024**2,2),'MiB',flush=True)
    write_json(root/'runs.json',[index[key] for key in sorted(index)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUT)
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args();package(args.root,args.names)
