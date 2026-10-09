"""Read-only F06/F07 actual evidence, source-helix metrics and correct videos."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,shutil,importlib.util
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.full_source_inner_experiment import BASE
from validation.suspended_supply_experiment import write
from validation.hook_start_transfer_experiment import render
from snake_control.suspended_supply_transfer import nodes
from snake_control.head_first_inner_route import HeadFirstInnerRoute

def source_stats(c):
    r=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);phi=np.unwrap(np.arctan2(c[:,2]-1,c[:,1]));f=np.polyfit(phi,c[:,0],1)
    return dict(radius_mm=r*1000,pitch_mm=float(f[0]*2*np.pi*1000),axis_fit_rms_mm=float(np.std(c[:,0]-np.polyval(f,phi))*1000),wrap_deg=float(np.degrees(np.ptp(phi))),near_source_rods=int(np.count_nonzero((r>.075)&(r<.115))))

def review(name):
    case=BASE/name;z=np.load(case/'trace.npz');t=z['sim_time'];p=z['points_mm']/1000;c=z['centers_mm']/1000;f=z['contact_force_N'];result=json.loads((case/'result.json').read_text());commands=json.loads((case/'commands.json').read_text())
    ids=[0,int(np.argmin(abs(t-14.7))),int(np.argmin(abs(t-24.7))),int(np.argmin(abs(t-111.65))),len(t)-1];stages=[]
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model/Snake_Robot_o20_full_source_inner.xml'));d=mujoco.MjData(m);fk=0.
    fig,axes=plt.subplots(3,len(ids),figsize=(19,11))
    for k,i in enumerate(ids):
        d.qpos[:]=z['qpos'][i];mujoco.mj_fwdPosition(m,d);fk=max(fk,float(np.linalg.norm(nodes(d)-p[i],axis=1).max()*1000));ss=source_stats(c[i]);entry=p[i,0]-p[i,1]
        stages.append(dict(t_s=float(t[i]),head_mm=p[i,0]*1000,head_entry_vector_mm=entry*1000,source_target_force_n=f[i].sum(1),source_bearing_modules=np.flatnonzero(f[i,0]>.1),source_shape=ss,actual_points_mm=p[i]*1000,reference_points_mm=z['reference_points_mm'][i]))
        for row,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            ax=axes[row,k];ax.plot(p[i,:,a],p[i,:,b],'o-',ms=2.5,label='Actual');ax.plot(z['reference_points_mm'][i,:,a]/1000,z['reference_points_mm'][i,:,b]/1000,'--',label='Executed guide');ax.scatter(p[i,0,a],p[i,0,b],c='red',label='Head')
            if row==0:ax.plot([-.2,1.1],[0,0],color='gray',lw=8,alpha=.25);ax.plot([1.1,1.1],[0,2],color='gray',lw=8,alpha=.25);ax.set(xlim=(.35,1.3),ylim=(-.25,.55))
            elif row==1:ax.axhline(1,color='gray',ls=':');ax.axvline(1.1,color='gray',ls=':');ax.set(xlim=(.35,1.3),ylim=(.70,1.2))
            else:
                th=np.linspace(0,2*np.pi,200);ax.fill(.0565*np.cos(th),1+.0565*np.sin(th),color='gray',alpha=.25);ax.set(xlim=(-.25,.25),ylim=(.72,1.2))
            ax.set(title=f't={t[i]:.2f}s\nsource pitch fit {ss["pitch_mm"]:.1f}mm',xlabel='World '+['X','X','Y'][row]+'(m)',ylabel='World '+['Y','Z UP','Z UP'][row]+'(m)',aspect='equal');ax.legend(fontsize=6)
    fig.suptitle(name+': actual approach and unstable positioning; no target acquisition');fig.tight_layout();fig.savefig(case/'actual_stages.png',dpi=145);plt.close(fig)
    head_error=np.linalg.norm(p[:,0]-z['reference_points_mm'][:,0]/1000,axis=1)*1000
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True);axes[0].plot(t,p[:,0,0]*1000,label='Actual head X');axes[0].plot(t,p[:,0,1]*1000,label='Actual head Y');axes[0].plot(t,(p[:,0,2]-1)*1000,label='Actual head Z-axis offset');axes[0].plot(t,z['shift_mm'],label='Command material coordinate (not axial retreat)');axes[0].legend();axes[0].set(ylabel='mm')
    axes[1].plot(t,f[:,0].sum(1),label='Actual source bearing');axes[1].plot(t,f[:,1].sum(1),label='Actual target bearing');axes[1].legend();axes[1].set(ylabel='N')
    axes[2].plot(t,head_error,label='World guide head error');axes[2].plot([x['t_s'] for x in commands],[x['tracking_deg'] for x in commands],label='Drive tracking deg');axes[2].plot(t,z['pipe_depth_mm'],label='Actual pipe depth mm');axes[2].legend();axes[2].set(xlabel='Actual physical time(s)');fig.tight_layout();fig.savefig(case/'actual_progress.png',dpi=150);plt.close(fig)
    review=dict(name=name,physical_steps_in_review=0,outcome=result['outcome'],whole_transfer_complete=False,initial_all21_source=True,initial_source_shape=source_stats(c[0]),initial_target_force_n=float(f[0,1].sum()),first_target_contact_s=None if not (f[:,1].sum(1)>.2).any() else float(t[np.flatnonzero(f[:,1].sum(1)>.2)[0]]),maximum_target_force_n=float(f[:,1].sum(1).max()),actual_seconds=float(t[-1]),actual_head_min_x_mm=float(p[:,0,0].min()*1000),actual_head_max_x_mm=float(p[:,0,0].max()*1000),maximum_world_head_error_mm=float(head_error.max()),maximum_tracking_deg=max(x['tracking_deg'] for x in commands),frozen_model_fk_node_error_mm=fk,stage_metrics=stages,physical_root_reset_during_run=False,yamada_compliance_enabled=False)
    write(case/'actual_review.json',review)
    arrays={key:z[key] for key in z.files}
    rows=[{key:a[i] for key,a in arrays.items() if a.shape and a.shape[0]==len(t)} for i in range(len(t))]
    render(case,rows,model_name='Snake_Robot_o20_full_source_inner.xml',initial_label='Full source head-first start')
    dest=case/'review_sources/validation/hook_start_transfer_experiment.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'validation/hook_start_transfer_experiment.py',dest)
    print(name,'review/video done',flush=True)

def rejected_curve():
    case=BASE/'F07';z=np.load(case/'trace.npz');i=int(np.argmin(abs(z['sim_time']-111.65)))
    spec=importlib.util.spec_from_file_location('frozen_f07_head_route',case/'sources/snake_control/head_first_inner_route.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    route=frozen.HeadFirstInnerRoute(z['points_mm'][i]/1000,smooth_source=False);v=np.linspace(route.knots[-1],route.total,3001);p=route.sample(v)
    write(case/'rejected_continuous_curve.json',dict(status='Rejected by private actual-exit preflight; zero physical feed along this route',route_source='F07/sources/snake_control/head_first_inner_route.py',seed_actual_time_s=float(z['sim_time'][i]),stations_mm=v*1000,points_mm=p*1000,bridge_length_mm=route.bridge_length*1000,target_join_mm=route.target_join*1000))
    fig=plt.figure(figsize=(13,8));ax=fig.add_subplot(121,projection='3d');ax.plot(*p.T,label='Proposed continuous exact-rod route');ax.plot(*(z['points_mm'][i]/1000).T,'o-',ms=3,label='Actual exit seed');ax.view_init(27,140);ax.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)');ax.legend()
    ax=fig.add_subplot(122);checks=json.loads((case/'actual_exit_route_preflight.json').read_text())['checks'];ax.plot([x['shift_mm'] for x in checks],[x['fit_rms_mm'] for x in checks],'o-',label='Independent initialization fit mm');ax.plot([x['shift_mm'] for x in checks],[x['pipe_depth_mm'] for x in checks],'o-',label='Private pipe depth mm');cont=json.loads((BASE/'POST_F07_CONTINUATION_UNRUN/checks.json').read_text());ax.plot([x['shift_mm'] for x in cont],[x['pipe_depth_mm'] for x in cont],label='Continued private fit pipe depth mm');ax.set(xlabel='Local feed shift(mm)',ylabel='mm',title='Fit remains unreliable; not global infeasibility proof');ax.legend();fig.tight_layout();fig.savefig(case/'rejected_route_and_fit.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    for name in ['F06','F07']:review(name)
    rejected_curve()
