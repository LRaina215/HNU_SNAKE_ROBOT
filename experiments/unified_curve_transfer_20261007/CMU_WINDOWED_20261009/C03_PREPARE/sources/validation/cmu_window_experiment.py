"""CMU2025 direct windowed joint waves from one artificial all-source start."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,shutil,traceback
from datetime import datetime,timezone
import numpy as np,mujoco,cv2
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.full_source_inner_experiment import IMODEL,freeze
from validation.inner_hook_transfer_experiment import InnerSim
from validation.hook_start_transfer_experiment import shape_stats
from validation.suspended_supply_experiment import write,default
from snake_control.suspended_supply_transfer import nodes,LINK,DT
from snake_control.cmu_window_gait import calibrate,placed_pose,CMUWindowControl,CENTRE_R,EMPIRICAL_BEND
BASE=ROOT/'experiments/unified_curve_transfer_20261007/CMU_WINDOWED_20261009'

class CMUSim(InnerSim):
    def snapshot(self):
        r=super().snapshot();r['window_transition_modules']=self.info.get('window_transition_modules',0.);r['loosening_factor']=self.info.get('loosening_factor',1.);return r

def selection(model,A,w,ph,bend):
    d=mujoco.MjData(model);target={model.geom(n).id for n in ['pipe_straight_04','pipe_sharp_out_03']};records=[]
    for x in np.arange(.95,.60,-.005):
        q,p,s=placed_pose(model,A,w,ph,bend,x);d.qpos[:]=q;mujoco.mj_forward(model,d);c=(p[:-1]+p[1:])/2
        tg=[ct for ct in d.contact if ({ct.geom1,ct.geom2}&target) and ct.dist<=0]
        e=np.array([1.1-CENTRE_R,max(CENTRE_R,p[bend,1]),1.]);gap=np.linalg.norm(e-p[bend]);margin=(bend-EMPIRICAL_BEND)*LINK-gap
        pd=max([-ct.dist*1000 for ct in d.contact if any((model.geom(g).name or '').startswith('pipe_') for g in [ct.geom1,ct.geom2])],default=0.)
        valid=not tg and c[:,0].max()<1.085 and c[:,0].min()>-.18 and margin>=0 and pd<1.5
        rec=dict(seam_x_m=x,initial_target_contacts=len(tg),initial_pipe_depth_mm=pd,gap_lower_bound_mm=gap*1000,release_length_margin_mm=margin*1000,admissible=valid);records.append(rec)
        if valid:return q,p,s,rec,records,e
    raise RuntimeError(f'No initial pose with sufficient release length for bend={bend}')

def charts(case,sim,ctrl,p,e,geometry,rows=None):
    initial_fk=mujoco.MjData(sim.m);initial_fk.qpos[:]=np.load(case/'initial_pose.npz')['qpos'];mujoco.mj_fwdPosition(sim.m,initial_fk)
    b=ctrl.bend;fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,inds,title in [(axes[0],[0,1],'Top XY / inner angle'),(axes[1],[0,2],'Target XZ / upward is +Z')]:
        ax.plot(p[:,inds[0]],p[:,inds[1]],'.-',color='navy',label='Initial all-source FK')
        ax.plot(p[:b+1,inds[0]],p[:b+1,inds[1]],'o-',color='darkorange',label=f'Release {b} rods / {b*LINK*1000:.0f}mm')
        ax.plot(p[b:,inds[0]],p[b:,inds[1]],'o-',color='forestgreen',label=f'Posterior {21-b} rods / {(21-b)*LINK*1000:.1f}mm')
        ax.scatter(*e[inds],marker='x',c='crimson',label='Desired target inner entry')
        ax.plot([p[b,inds[0]],e[inds[0]]],[p[b,inds[1]],e[inds[1]]],'--',color='crimson',label='Distance lower bound, not a path')
        ax.annotate('Exit N'+str(b),p[b,inds],xytext=(4,10),textcoords='offset points');ax.annotate('Head',p[0,inds],xytext=(4,8),textcoords='offset points')
        ax.set(title=title,xlabel='World X(m)',ylabel='World Y(m)' if inds[1]==1 else 'World Z(m)',aspect='equal');ax.legend(fontsize=7)
    theta=np.linspace(0,2*np.pi,201);axes[1].fill(1.1+.0565*np.cos(theta),1+.0565*np.sin(theta),color='gray',alpha=.3);axes[0].plot([.45,1.1],[0,0],lw=15,c='gray',alpha=.3);axes[0].plot([1.1,1.1],[0,.3],lw=15,c='gray',alpha=.3)
    ax=axes[2];ax.axis('off');ax.text(0,.95,f'J1..J{b}: first {b} bending hinges\nM0..M{b-1}: release rods\nN{b}: selected release seam\nM{b}: selected upward module\n\nPaper empirical target wrap: {EMPIRICAL_BEND} rods\nFront length: {b*LINK*1000:.0f}mm\nGap lower bound: {geometry["gap_lower_bound_mm"]:.1f}mm\nLength margin: {geometry["release_length_margin_mm"]:.1f}mm\n\nGeometric length adequacy is necessary,\nnot evidence of collision-free reach.\nInitial target contact: absent\nOriginal dynamics retained',va='top',fontsize=11)
    fig.tight_layout();fig.savefig(case/'selection_release_length.png',dpi=160);plt.close(fig)
    # Direct wave FK at every stage, rigidly anchored at the initial selected
    # module frame. This is a nominal reference, not executed motion.
    fig=plt.figure(figsize=(16,10));ref=[]
    for i,nt in enumerate([0,2,b/2,b,b+4,20]):
        t=nt/ctrl.window_rate;q,n,L,mask,phase=ctrl.reference(t);pp=ctrl.forward_reference(initial_fk,q,b);ax=fig.add_subplot(2,3,i+1,projection='3d');ax.plot(*pp.T,'o-',ms=3,c='purple');ax.plot(*p.T,'--',c='gray',alpha=.4)
        ax.plot([.45,1.1],[0,0],[1,1],c='gray',lw=8,alpha=.25);ax.plot([1.1,1.1],[0,.5],[1,1],c='gray',lw=8,alpha=.25);ax.set(title=f'Nominal FK nt={nt:g}, L={L:.3f}',xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)');ax.view_init(27,140);ref.append(dict(t_s=t,nt=nt,loosening=L,phase=phase,q_rad=q,points_mm=pp*1000))
    fig.suptitle('Complete direct-wave reference / initial M'+str(b)+' frame held for illustration / NOT actual motion');fig.tight_layout();fig.savefig(case/'complete_wave_fk_reference.png',dpi=150);plt.close(fig);write(case/'complete_wave_fk_reference.json',ref)
    if rows:
        fig=plt.figure(figsize=(16,5));rr=[]
        for i,j in enumerate([0,len(rows)//2,len(rows)-1]):
            r=rows[j];ax=fig.add_subplot(1,3,i+1,projection='3d');a=r['points_mm']/1000;f=r['reference_points_mm']/1000;ax.plot(*a.T,'o-',ms=3,label='Actual physics FK');ax.plot(*f.T,'--',label='Dispatched wave FK / actual carrier frame');ax.set(title=f'Actual {r["sim_time"]:.2f}s',xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)');ax.view_init(27,140);ax.legend(fontsize=7);rr.append(dict(t_s=r['sim_time'],actual_mm=a*1000,command_fk_mm=f*1000))
        fig.tight_layout();fig.savefig(case/'reference_actual_comparison.png',dpi=160);plt.close(fig);write(case/'reference_actual_comparison.json',rr)

def video(case,rows):
    m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model'/IMODEL.name));d=mujoco.MjData(m);ren=mujoco.Renderer(m,height=600,width=900);cam=mujoco.MjvCamera();cam.lookat[:]=[.93,.20,.95];cam.distance=1.7;cam.azimuth=140;cam.elevation=-26
    wr=cv2.VideoWriter(str(case/'replay_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1260,600));ids=list(range(0,len(rows),5))
    if not wr.isOpened():raise RuntimeError('Video writer unavailable')
    for fi,j in enumerate(ids):
        r=rows[j];d.qpos[:]=r['qpos'];mujoco.mj_forward(m,d)
        zmin=float(r['points_mm'][:,2].min()/1000);drop=np.clip((.8-zmin)/.8,0,1);cam.lookat[2]=.95-.40*drop;cam.distance=1.7+.8*drop
        ren.update_scene(d,camera=cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);pan=np.full((600,360,3),25,np.uint8)
        cv2.putText(im,f'{case.name} ACTUAL continuous 1x {r["sim_time"]:.2f}s',(10,25),0,.65,(255,255,255),1)
        lines=['CMU window wave / all-source start',f'Window nt {r["window_transition_modules"]:.3f} (nominal)',f'Loosening factor {r["loosening_factor"]:.3f}',f'Actual source clear {r["actual_exit_mask"].sum()}/21',f'Actual target bearing rods {r["actual_received_mask"].sum()}/21',f'Source/target N {r["contact_force_N"].sum(1).round(1)}',f'Pipe/self mm {r["pipe_depth_mm"]:.2f}/{r["self_depth_mm"]:.2f}','Actual joints / Kp / drive Nm']
        for k,line in enumerate(lines):cv2.putText(pan,line,(8,24+k*24),0,.43,(225,225,225),1)
        for k in range(20):cv2.putText(pan,f'{k+1:02d} {r["kp_nm_per_rad"][k]:5.1f} {r["torque"][k]:+4.2f}',(8+(k//10)*176,230+(k%10)*32),0,.43,(220,225,180),1)
        frame=np.hstack([im,pan]);wr.write(frame)
        if fi in {0,len(ids)//2,len(ids)-1}:cv2.imwrite(str(case/f'frame_{fi:05d}.jpg'),frame)
    wr.release();ren.close();cap=cv2.VideoCapture(str(case/'replay_1x.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));checks=[]
    for j in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,j);ok,_=cap.read();checks.append(dict(frame=j,readable=ok))
    write(case/'video.json',dict(fps=cap.get(cv2.CAP_PROP_FPS),frames=n,actual_seconds=rows[-1]['sim_time'],speed=1.,source='Saved actual qpos from one continuous dynamics run; replay data assignments only; no I01 state',readback=checks));cap.release()

def run(args):
    case=BASE/args.name;case.mkdir(parents=True,exist_ok=False);m=mujoco.MjModel.from_xml_path(str(IMODEL));A,w,ph,_,cs=calibrate(m,radius=args.radius)
    q,p,cs,sel,scan,e=selection(m,A,w,ph,args.bend);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];initial=dict(qpos=q,qvel=np.zeros(m.nv),ctrl=q[qa]);sim=CMUSim(case,initial,model_path=IMODEL);r0=sim.snapshot();ctrl=CMUWindowControl(sim.m,sim.scratch,A,w,ph,args.bend,frequency=args.frequency,delta_sign=args.delta_sign)
    sr=np.linalg.norm(((p[:-1]+p[1:])/2)[:,[1,2]]-[0,1],axis=1);tangent=(p[args.bend]-p[args.bend+1])/LINK
    ready=bool(r0['pipe_depth_mm']<1.5 and r0['self_depth_mm']<.5 and not r0['ground'] and sel['initial_target_contacts']==0 and np.all((sr>.075)&(sr<.110)) and tangent[2]>.9 and abs(tangent[0])<.25 and abs(tangent[1])<.3)
    continuous_wrap_mm=.75*np.hypot(2*np.pi*cs['radius'],.180)*1000
    full_length_margin_mm=args.bend*LINK*1000-continuous_wrap_mm-sel['gap_lower_bound_mm']
    empirical_ready=ready;ready=ready and (full_length_margin_mm>=0 or args.empirical_length_only)
    geometry=dict(initial_state='Artificial entire native-sine source helix, zero velocity, no target contact; placement before first mj_step only',physical_steps=0,source_start_module=0,source_modules=21,initial_points_mm=p*1000,source_radius_mm=sr*1000,release_rods=args.bend,release_length_mm=args.bend*LINK*1000,posterior_support_rods=21-args.bend,posterior_support_length_mm=(21-args.bend)*LINK*1000,empirical_target_wrap_rods=EMPIRICAL_BEND,selected_upward_module=args.bend,selected_tail_to_head_vector=tangent,amplitude_rad=A,spatial_rad_per_hinge=w,source_phase_rad=ph,calibrated_circle=cs,initial_source_target_force_n=r0['contact_force_N'].sum(1),ready=ready,**sel)
    geometry.update(empirical_ready=empirical_ready,continuous_270deg_wrap_length_mm=continuous_wrap_mm,continuous_wrap_and_provisional_gap_margin_mm=full_length_margin_mm,empirical_length_only=args.empirical_length_only,length_guard_scope='Reject insufficient exact270deg pitched-wrap length; positive margin alone does not prove a collision-free lower-entry route')
    write(case/'initial_geometry.json',geometry);write(case/'selection_candidates.json',scan);np.savez(case/'initial_pose.npz',**initial);mujoco.mj_getState(sim.m,sim.d,sim.state,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez(case/'initial_state.npz',integration_state=sim.state,sim_time=0.)
    freeze(case)
    for f in ['snake_control/cmu_window_gait.py','validation/cmu_window_experiment.py']:
        dst=case/'sources'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    write(case/'plan.json',dict(name=args.name,registered_utc=datetime.now(timezone.utc).isoformat(),maximum_trials=3,method='CMU equations8/9/12 direct alternating-axis sine waves; no world-curve inverse fit',bend=args.bend,window_phase_delta_rad=ctrl.delta,phase_delta_reason='C01 literal negative sign; C02 positive sign because native FK expands selected upward M8 toward target only for head phase -pi/2 relative to source; L-pipe branch adaptation, not identical paper convention',frequency_cycles_s=args.frequency,phase_rate_rad_s=ctrl.omega,window_rate_modules_s=ctrl.window_rate,equation10_interpretation='Per-cycle travel pi*d_module; nt_dot=pi*d_module*f/link; printed inverse-speed equation is dimensionally inconsistent',settle_seconds=args.settle,maximum_actual_seconds=args.seconds,maximum_wall_seconds=args.max_wall,deadline=args.deadline,physics='original gravity, friction, damping1.5, Kp127, torque2.355Nm, actual90deg and command88deg',stop='Ground; actual joint>90.5deg; self>3mm; pipe>3mm sustained0.2s; nonfinite; acquired grip falls1s; all-window without strict shape30s; deadline',completion='Actual all21 single target coil, original source clear, posterior target force>0.2N, allKp127, hold1s',physical_root_reset_during_run=False,initial_state_is_artificial=True,yamada_compliance_enabled=False))
    charts(case,sim,ctrl,p,e,geometry);print(json.dumps(dict(name=args.name,ready=ready,**sel),default=default),flush=True)
    if not ready:write(case/'result.json',dict(outcome='initial_selection_not_ready',physical_steps=0));return
    start=time.monotonic();deadline=datetime.fromisoformat(args.deadline).timestamp();outcome='maximum_actual_duration';deep=fall=0.;acquired=False;since=None;hold=0.;all_since=None;milestones={};error=None
    try:
        while sim.d.time<args.seconds:
            t=float(sim.d.time);elapsed=max(0,t-args.settle)
            if (case/'stop_and_save.request').exists():outcome='requested_stop';break
            if time.monotonic()-start>args.max_wall or time.time()>deadline:outcome='execution_wall_budget';break
            r=sim.rows[-1]
            if t<args.settle:target=initial['ctrl'];info=dict(stage='settle_source',window_transition_modules=0.,loosening_factor=1.,maximum_dispatch_step_deg=0.,reference_points_mm=p*1000)
            else:target,info=ctrl.command(sim.scratch,elapsed,r['contact_force_N'],r['source_contact_any'])
            sim.info=info;sim.commands.append(dict(t_s=t,ctrl_rad=target.copy(),**info));sim.command(target);r=sim.rows[-1];stats=shape_stats(r,1)
            if r['ground']:outcome='ground_contact';break
            if abs(r['q']).max()>np.radians(90.5):outcome='actual_joint_limit';break
            if r['self_depth_mm']>3:outcome='actual_self_interference';break
            deep=deep+DT if r['pipe_depth_mm']>3 else 0.
            if deep>=.2:outcome='actual_pipe_interference';break
            if not np.isfinite(sim.d.qpos).all():outcome='nonfinite';break
            c=r['centers_mm']/1000;rr=np.linalg.norm(c[:args.bend,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(c[:args.bend,2]-1,c[:args.bend,0]-1.1));wrap=np.degrees(np.ptp(phi));bearing=r['contact_force_N'][1,:args.bend].sum()
            if bearing>.2 and 'first_target_bearing' not in milestones:milestones['first_target_bearing']=dict(t_s=r['sim_time'],head_mm=r['points_mm'][0])
            if np.count_nonzero((rr>.065)&(rr<.125)&(c[:args.bend,1]>.15))>=EMPIRICAL_BEND and wrap>180 and bearing>1:
                if not acquired:acquired=True;milestones['front_acquisition']=dict(t_s=r['sim_time'],wrap_deg=wrap,bearing_n=bearing)
            if acquired:
                falling=bearing<.1 and (np.median(rr)>.145 or c[:args.bend,2].mean()<.9);fall=fall+DT if falling else 0.
                if fall>=1:outcome='acquired_grip_collapsed_falling';break
            full=ctrl.window_rate*elapsed>=20
            if full and all_since is None:all_since=r['sim_time']
            complete=full and stats['strict_actual_shape'] and not r['source_contact_any'].any() and r['contact_force_N'][0].sum()<.1 and r['contact_force_N'][1,18:].sum()>.2
            if complete and since is None:since=r['sim_time']
            if not complete:since=None
            hold=0 if since is None else r['sim_time']-since
            if hold>=1:outcome='complete_cmu_window_from_all_source';break
            if full and r['sim_time']-all_since>30:outcome='whole_wave_without_strict_actual_shape_30s';break
            if len(sim.commands)%100==0:
                msg=dict(t_s=round(r['sim_time'],2),nt=r['window_transition_modules'],source_target_force_n=r['contact_force_N'].sum(1),head_mm=r['points_mm'][0],pipe_mm=r['pipe_depth_mm'],front_wrap_deg=wrap,acquired=acquired);print(json.dumps(msg,default=default),flush=True);write(case/'progress.json',msg)
    except Exception:outcome='implementation_exception';error=traceback.format_exc();print(error,flush=True)
    finally:
        errors=sim.save();r=sim.rows[-1];stats=shape_stats(r,1)
        result=dict(name=args.name,outcome=outcome,actual_seconds=r['sim_time'],wall_seconds=time.monotonic()-start,physical_steps=int(round(sim.d.time/sim.m.opt.timestep)),initial_all_source=ready,active_front_acquisition=acquired,whole_transfer_complete=outcome=='complete_cmu_window_from_all_source',milestones=milestones,final_source_target_force_n=r['contact_force_N'].sum(1),final_source_contact_modules=np.flatnonzero(r['source_contact_any']),window_nt=r['window_transition_modules'],actual_head_displacement_mm=r['points_mm'][0]-r0['points_mm'][0],maximum_actual_joint_deg=float(np.degrees(max(abs(x['q']).max() for x in sim.rows))),maximum_drive_torque_nm=float(max(abs(x['torque']).max() for x in sim.rows)),maximum_command_step_deg=max((x['maximum_dispatch_step_deg'] for x in sim.commands),default=0),maximum_pipe_depth_mm=max(x['pipe_depth_mm'] for x in sim.rows),maximum_self_depth_mm=max(x['self_depth_mm'] for x in sim.rows),completion_hold_seconds=hold,physical_root_reset_during_run=False,initial_state_is_artificial=True,yamada_compliance_enabled=False,save_errors=errors,error=error,**stats)
        write(case/'result.json',result);write(case/'controller_state.json',dict(last_ctrl=ctrl.last,kp=ctrl.kp,phase=ctrl.phase,window_rate=ctrl.window_rate));print(json.dumps(result,default=default),flush=True)
        charts(case,sim,ctrl,p,e,geometry,sim.rows)
        try:video(case,sim.rows)
        except Exception:write(case/'video_error.json',dict(error=traceback.format_exc()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='C01');ap.add_argument('--bend',type=int,default=8);ap.add_argument('--radius',type=float,default=.088);ap.add_argument('--frequency',type=float,default=.02);ap.add_argument('--delta-sign',type=int,choices=[-1,1],default=-1);ap.add_argument('--empirical-length-only',action='store_true',help='Legacy C01/C02 empirical admission; does not establish full path length');ap.add_argument('--settle',type=float,default=.5);ap.add_argument('--seconds',type=float,default=450);ap.add_argument('--max-wall',type=float,default=800);ap.add_argument('--deadline',default='2026-10-09T12:56:59+00:00');run(ap.parse_args())
