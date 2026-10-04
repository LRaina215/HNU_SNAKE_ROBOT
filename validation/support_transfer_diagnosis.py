"""Inspect energy and commands around recorded support loss; no new physics."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from three_segment_review import context
from segmented_pole_experiment import write_json


def diagnose(folder):
    result,z,m,d,meta=context(folder)
    t=z['time']-z['time'][0]
    absent=~np.any(z['normal_force'][:,0]>.1,axis=1)
    sustained=np.flatnonzero(np.convolve(absent.astype(int),np.ones(60),mode='full')[:len(t)]>=60)
    onset=float(t[sustained[0]-59]) if len(sustained) else float(t[-1])
    selected=np.flatnonzero((t>=max(1.,onset-2))&(t<=onset+2))
    # Enable energy calculation on this offline model copy only.
    m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    energy=[]
    for i in selected:
        d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i]
        mujoco.mj_forward(m,d);energy.append(d.energy.copy())
    energy=np.asarray(energy)
    dofs=m.jnt_dofadr[m.actuator_trnid[:,0]]
    power=np.sum(z['torque'][selected]*z['qvel'][selected][:,dofs],axis=1)
    work=np.r_[0,np.cumsum((power[1:]+power[:-1])*.5*np.diff(t[selected]))]
    centers=(z['points_mm'][:,:-1]+z['points_mm'][:,1:])/2
    speed=np.r_[np.zeros((1,21)),np.linalg.norm(np.diff(centers,axis=0),axis=2)/np.diff(t)[:,None]]
    i,j=np.unravel_index(np.where((t>=1)[:,None],speed,-1).argmax(),speed.shape)
    report=dict(first_sustained_source_loss_onset_seconds=onset,
        observation_window_seconds=[float(t[selected[0]]),float(t[selected[-1]])],
        potential_change_J=float(energy[-1,0]-energy[0,0]),
        maximum_kinetic_J=float(energy[:,1].max()),motor_work_J=float(work[-1]),
        maximum_module_speed=dict(time_seconds=float(t[i]),module=int(j),speed_mm_s=float(speed[i,j])),
        joint_position_gain=m.actuator_gainprm[:,0].tolist(),
        torque_limit=m.actuator_forcerange[:,1].tolist(),
        note='Potential/kinetic energy and actuator work are diagnostics; this does not isolate friction, contact dissipation or a unique causal mechanism.')
    write_json(folder/'support_diagnosis.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=folder.parent/'reviews';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].plot(t[selected],energy[:,0]-energy[0,0],label='Potential energy change')
    axes[0].plot(t[selected],energy[:,1],label='Kinetic energy')
    axes[0].plot(t[selected],work,label='Actuator work')
    for k,label in ((0,'Source normal force'),(1,'Target normal force')):
        axes[1].plot(t[selected],z['normal_force'][selected,k].sum(axis=1),label=label)
    axes[2].plot(t[selected],speed[selected,20],label='Tail module speed')
    for ax,label in zip(axes,['J','N','mm/s']):
        ax.set_ylabel(label);ax.axvline(onset,color='grey',ls='--');ax.grid(alpha=.3);ax.legend()
    axes[-1].set_xlabel('Physical seconds after checkpoint')
    fig.suptitle(folder.name+' | first sustained departure from source pipe')
    fig.tight_layout();fig.savefig(out/(folder.name+'_support_energy.png'),dpi=130);plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    args=p.parse_args();print(diagnose(args.folder))
