"""Read saved actual traces; create geometry audit and desired target FK."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.cmu_window_experiment import BASE,IMODEL,video
from validation.hook_start_transfer_experiment import shape_stats
from validation.suspended_supply_experiment import write,default
from snake_control.cmu_window_gait import wave,cylinder
from snake_control.suspended_supply_transfer import nodes

def review():
    reports=[]
    for name in ['C01','C02']:
        case=BASE/name;z=np.load(case/'trace.npz');t=z['sim_time'];g=json.loads((case/'initial_geometry.json').read_text(encoding='utf-8'));p=json.loads((case/'plan.json').read_text(encoding='utf-8'));r=json.loads((case/'result.json').read_text(encoding='utf-8'));cmd=json.loads((case/'commands.json').read_text(encoding='utf-8'))
        target=z['contact_force_N'][:,1].sum(1);front=z['contact_force_N'][:,1,:8].sum(1);i=np.flatnonzero(target>.2);j=np.flatnonzero(front>.2)
        req=.75*np.hypot(2*np.pi*g['calibrated_circle']['radius'],.180)*1000;gap=g['gap_lower_bound_mm'];margin=g['release_length_mm']-req-gap
        audit=dict(name=name,trace_samples=len(t),trace_interval_s=[float(np.diff(t).min()),float(np.diff(t).max())],first_any_target_side_bearing_gt_point2_n_s=float(t[i[0]]) if len(i) else None,first_front8_target_side_bearing_gt_point2_n_s=float(t[j[0]]) if len(j) else None,initial_target_contact_count=g['initial_target_contacts'],kp_range=float(z['kp_nm_per_rad'].min()),maximum_tracking_deg=float(np.degrees(abs(z['q']-z['ctrl'])).max()),target_wrap_270deg_helix_length_mm=req,gap_lower_bound_mm=gap,front_release_length_mm=g['release_length_mm'],continuous_270deg_wrap_plus_gap_margin_mm=margin,empirical_six_rods_plus_gap_margin_mm=g['release_length_margin_mm'],selection_scope='Empirical six-rod budget plus Euclidean gap only; neither sufficient proof of an exact270deg pitched wrap nor of a collision-free path',outcome=r['outcome'],phase_delta_rad=p.get('window_phase_delta_rad',-np.pi/2))
        illustrative_entry=np.array([1.1+.088*np.cos(5*np.pi/4),.16,1+.088*np.sin(5*np.pi/4)]);entry_gap=np.linalg.norm(illustrative_entry-np.asarray(g['initial_points_mm'])[g['release_rods']]/1000)*1000
        audit.update(illustrative_lower_entry_mm=illustrative_entry*1000,illustrative_lower_entry_straight_gap_mm=entry_gap,illustrative_lower_entry_plus_270deg_wrap_margin_mm=g['release_length_mm']-entry_gap-req,gap_scope='118.8mm is distance to a provisional inner-side point at axis height, not the later illustrated lower entry, and not a global reachability bound')
        write(case/'evidence_audit.json',audit);reports.append(audit)
        m=mujoco.MjModel.from_xml_path(str(case/'sources/mujoco_model'/IMODEL.name));d=mujoco.MjData(m);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=wave(g['amplitude_rad'],g['spatial_rad_per_hinge'],g['source_phase_rad']);mujoco.mj_fwdPosition(m,d);pp=nodes(d);cs=cylinder(pp);rot=Rotation.align_vectors([[0.,1,0]],[cs['axis']])[0].as_matrix();translation=np.array([1.1,.55,1])-rot@cs['origin'];d.qpos[:3]=translation;d.qpos[3:7]=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]];mujoco.mj_fwdPosition(m,d);pp=nodes(d);cc=(pp[:-1]+pp[1:])/2;stat=shape_stats(dict(centers_mm=cc*1000,contact_force_N=np.zeros((2,21))),1)
        write(case/'desired_all_target_fk.json',dict(status='Desired single target helix, privately rigid-placed full-wave FK; NOT simulated or executed',qpos=d.qpos,points_mm=pp*1000,axis=[0,1,0],circle=cs,**stat))
        initial=np.asarray(g['initial_points_mm'])/1000;seam=initial[g['release_rods']];s=np.linspace(0,.75,301);phi=5*np.pi/4-2*np.pi*s;desired=np.column_stack([1.1+.088*np.cos(phi),.16+.18*s,1+.088*np.sin(phi)])
        fig=plt.figure(figsize=(16,10))
        ax=fig.add_subplot(221,projection='3d');ax.plot(*initial.T,'o-',ms=3,label='Initial all-source FK');ax.plot(*desired.T,c='green',label='Desired inner-lower -> upward270deg wrap');ax.plot(*np.vstack([seam,desired[0]]).T,'--',c='orange',label='Straight gap illustration / not safe motion');ax.scatter(*initial[0],c='red',label='Initial head');ax.set(title='Selection and desired wrap / length adequacy remains unproved',xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)');ax.view_init(27,140);ax.legend(fontsize=7)
        ax=fig.add_subplot(222,projection='3d');ax.plot(*pp.T,'o-',ms=3,c='green',label='Desired target-only FK');ax.plot([1.1,1.1],[.2,.9],[1,1],lw=9,c='gray',alpha=.3);ax.set(title='Desired final body / privately placed FK / NOT actual',xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)');ax.view_init(27,140);ax.legend(fontsize=7)
        ax=fig.add_subplot(223);head=z['points_mm'][:,0];ax.plot(head[:,0],head[:,2],label='Actual head XZ');ax.scatter(head[0,0],head[0,2],c='green',label='Start');ax.scatter(head[-1,0],head[-1,2],c='red',label='End');th=np.linspace(0,2*np.pi,201);ax.fill(1100+56.5*np.cos(th),1000+56.5*np.sin(th),c='gray',alpha=.3);ax.set(title='Actual continuous head trajectory',xlabel='World X(mm)',ylabel='World Z(mm)',aspect='equal');ax.legend(fontsize=7)
        ax=fig.add_subplot(224);ax.plot(t,z['contact_force_N'][:,0].sum(1),label='Actual source side bearing');ax.plot(t,target,label='Actual target side bearing');ax.plot(t,front,label='Actual front8 target side bearing');ax.set(title='Side bearing only / contact cap normals excluded',xlabel='Actual time(s)',ylabel='N');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(case/'desired_reference_actual_evidence.png',dpi=150);plt.close(fig)
    write(BASE/'evidence_summary.json',reports);print(json.dumps(reports,ensure_ascii=False,indent=2,default=default),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render-fall',action='store_true');args=ap.parse_args();review()
    if args.render_fall:
        case=BASE/'C02';archive=np.load(case/'trace.npz');z={k:archive[k] for k in archive.files};rows=[{k:v[i] for k,v in z.items()} for i in range(len(z['sim_time']))];video(case,rows);write(case/'video_render_revision.json',dict(reason='Camera follows falling robot height so the ground stop remains visible',dynamics_rerun=False,source='validation/cmu_window_experiment.py video camera revision; original actual trace unchanged'))
