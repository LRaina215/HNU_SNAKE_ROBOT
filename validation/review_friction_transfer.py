"""Compare saved actual runs and reconstruct contact slip without dynamics."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from snake_control.unified_curve_transfer import cylindrical_contact


def read_case(case):
    result=json.loads((case/'result.json').read_text(encoding='utf8'))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
    d=mujoco.MjData(m);head=m.body('head').id
    source={m.geom(x).id for x in ['pipe_straight_01','pipe_sharp_in_02']}
    target={m.geom(x).id for x in ['pipe_sharp_out_03','pipe_straight_04']}
    with np.load(case/'trace.npz') as z:
        indices=np.unique(np.r_[np.arange(0,len(z['sim_time']),10),len(z['sim_time'])-1])
        t=z['sim_time'][indices]-z['sim_time'][0]
        centers=z['centers_mm'][indices]
        force=z['cylindrical_contact_force_N'][indices]
        shift=z['shift_mm'][indices]
        qpos=z['qpos'][indices];qvel=z['qvel'][indices]
    slip=np.zeros_like(force)
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    for row in range(len(indices)):
        d.qpos[:]=qpos[row];d.qvel[:]=qvel[row]
        mujoco.mj_fwdPosition(m,d)
        count=np.zeros((2,21))
        for c in d.contact:
            for pipe,robot in [(c.geom1,c.geom2),(c.geom2,c.geom1)]:
                body=int(m.geom_bodyid[robot])
                side=0 if pipe in source else 1 if pipe in target else None
                if side is None or not head<=body<head+21 or not cylindrical_contact(side,c.pos,c.frame[:3]):
                    continue
                module=body-head
                if force[row,side,module]<=.1:
                    continue
                mujoco.mj_jac(m,d,jp,jr,c.pos,body)
                velocity=jp@d.qvel;normal=c.frame[:3]
                slip[row,side,module]+=np.linalg.norm(velocity-normal*(normal@velocity))*1000
                count[side,module]+=1
        slip[row]/=np.maximum(count,1.)
    numerator=(slip*force).sum(2);denominator=force.sum(2)
    average=numerator/np.maximum(denominator,1e-12)
    overall=np.trapz(numerator,t,axis=0)/np.maximum(np.trapz(denominator,t,axis=0),1e-12)
    report={key:result.get(key) for key in ['name','outcome','physical_seconds','complete','final_shift_mm',
        'new_sustained_target_modules','fixed_front_M0_M6_mean_y_advance_mm','fixed_rear_M14_M20_mean_x_advance_mm',
        'max_actual_joint_deg','max_actuator_torque_Nm','max_dispatch_step_deg','first_ctrl_error_rad',
        'runtime_effective_sliding_friction','contact_tangent_constraint_scale','save_errors']}
    report['reconstructed_load_time_weighted_sliding_mm_s']=overall.tolist()
    report['maximum_reconstructed_supported_point_sliding_mm_s']=slip.max(axis=(0,2)).tolist()
    report['sliding_method']='saved actual qpos/qvel; exact contact-point Jacobians; geometric contact points averaged within each module, then weighted by saved actual module cylinder load; 0.1s samples, no physics steps'
    np.savetxt(case/'contact_sliding_review.csv',np.column_stack([t,average]),delimiter=',',header='elapsed_s,source_load_weighted_sliding_mm_s,target_load_weighted_sliding_mm_s',comments='')
    return report,dict(t=t,front=(centers[:,:7,1]-centers[0,:7,1]).mean(1),
        rear=(centers[:,14:,0]-centers[0,14:,0]).mean(1),shift=shift,
        slip=average,contacts=np.count_nonzero(force>.1,axis=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=True)
    reports=[];fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for case in args.cases:
        report,data=read_case(case);reports.append(report)
        line,=axes[0,0].plot(data['t'],data['front'],label=case.name)
        color=line.get_color()
        axes[0,1].plot(data['t'],data['rear'],label=case.name,color=color)
        for side,style in [(0,'-'),(1,'--')]:
            label=f'{case.name} '+('source' if side==0 else 'target')
            axes[1,0].plot(data['t'],data['slip'][:,side],label=label,color=color,linestyle=style)
            axes[1,1].plot(data['t'],data['contacts'][:,side],label=label,color=color,linestyle=style)
    titles=['Actual front M0-M6 mean +Y','Actual rear M14-M20 mean +X','Actual relative contact sliding','Actual cylinder contact count']
    units=['mm','mm','mm/s','count']
    for ax,title,unit in zip(axes.ravel(),titles,units):
        ax.set_title(title);ax.set_xlabel('Actual elapsed time / s');ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Sliding friction and controller tangential-constraint comparisons')
    fig.savefig(args.out/'comparison.png',dpi=150)
    (args.out/'comparison.json').write_text(json.dumps(dict(evidence='continuous actual runs from same CLOSE1B; nominal shift is not physical movement',cases=reports),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(reports))
