"""Close incomplete geometry with bounded saved-state diagnostics, no physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import ast,csv,json,shutil
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from front_local_rolling_geometry import BASE,PARENT,ORIGINAL,write
from curve_local_path import LocalPathFit
from curve_guided_geometry import points
from snake_control.front_local_rolling import FrontLocalRollingReference


def ledger(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    result=json.loads((BASE/'geometry_result.json').read_text(encoding='utf8'))
    if result['status']!='NOT_READY':raise RuntimeError('This closure only records incomplete geometry')
    if ledger(PARENT/'trial_ledger.csv')!=ledger(BASE/'historical_17_trial_ledger.csv') or ledger(BASE/'trial_ledger.csv'):
        raise RuntimeError('Expected zero new dynamics and exact preserved historical rows')
    fit=LocalPathFit();ref=FrontLocalRollingReference(BASE/'reference_partial.npz',ORIGINAL)
    with np.load(BASE/'reference_partial.npz') as z:phases=z['phase_rad'].copy()
    initial=ref.state(0.);fit.set(initial);p0=points(fit.m,fit.d).copy()
    rows=[];first_collision=None;minimum=1e6;length=rear=0.;maxq=0.
    for phi in np.unique(np.r_[np.arange(0,phases[-1]+1e-8,np.radians(.05)),phases]):
        x=ref.state(float(phi));fit.set(x);mujoco.mj_forward(fit.m,fit.d);p=points(fit.m,fit.d)
        negative=[dict(geom1=fit.m.geom(c.geom1).name,geom2=fit.m.geom(c.geom2).name,distance_mm=float(c.dist*1000)) for c in fit.d.contact if c.dist<0]
        gap=float(fit.gaps(x)[:24].min());minimum=min(minimum,gap)
        length=max(length,float(abs(np.linalg.norm(np.diff(p,axis=0),axis=1)-64.5).max()))
        rear=max(rear,float(abs(p[12:]-p0[12:]).max()));maxq=max(maxq,float(np.degrees(abs(x[6:])).max()))
        row=dict(phase_deg=float(np.degrees(phi)),minimum_modifiable_gap_mm=gap,collisions=negative)
        if negative and first_collision is None:first_collision=row
        rows.append(row)
    write(BASE/'saved_partial_diagnostics.json',dict(physics_steps=0,checked_phase_range_deg=[0,float(np.degrees(phases[-1]))],
        interpolation_sampling_deg=.05,rows=rows,first_actual_finite_collision=first_collision,
        minimum_modifiable_conservative_gap_mm=minimum,maximum_length_error_mm=length,
        rear_nominal_node_maximum_error_mm=rear,maximum_joint_deg=maxq,
        interpretation='Saved failed prefix only. No full path, rate schedule, interface admission, actual holding or rolling result.'))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for phi in [0,phases[-1]]:
        x=ref.state(float(phi));fit.set(x);p=points(fit.m,fit.d)
        axes[0].plot(p[:,0],p[:,1],'.-',label=f'nominal phi={np.degrees(phi):.0f} deg')
        axes[1].plot(p[:8,0]-1100,p[:8,2]-1000,'.-',label=f'nominal phi={np.degrees(phi):.0f} deg')
    circle=np.linspace(0,2*np.pi,150);axes[1].plot(56.5*np.cos(circle),56.5*np.sin(circle),color='g',label='target pipe')
    axes[0].plot([650,1100],[0,0],'g-',lw=6);axes[0].plot([1100,1100],[0,450],'g-',lw=6)
    for ax in axes:ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend()
    axes[0].set_xlabel('World X (mm)');axes[0].set_ylabel('World Y (mm)')
    axes[1].set_xlabel('Target radial X (mm)');axes[1].set_ylabel('Target radial Z (mm)')
    fig.suptitle('OFFLINE failed geometry prefix only - NO DYNAMICS OR GRASP EVIDENCE');fig.tight_layout()
    fig.savefig(BASE/'offline_failed_prefix.png',dpi=140);plt.close(fig)
    sources=BASE/'sources';sources.mkdir(exist_ok=True)
    names=['snake_control/front_local_rolling.py','validation/front_local_rolling_geometry.py','validation/front_local_rolling_fit.py',
        'validation/front_local_rolling_interface.py','validation/front_local_rolling_experiment.py','validation/front_local_rolling_finalize.py',
        'snake_control/executable_grasp_state.py','snake_control/local_material_reference.py','mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml']
    for name in names:
        dest=sources/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf8'))
    registry=json.loads((BASE/'revision_registry.json').read_text(encoding='utf8'))
    registry.update(status='STOPPED_GEOMETRY_INCOMPLETE',new_dynamics=0,total_dynamics=17,further_dynamics_authorized=False,
        remaining_unused_trials=4,holding='NOT_TESTED_NEW_CANDIDATE',actual_rolling='NOT_TESTED',freeze='NOT_RUN',
        reason='Last permitted finite construction did not satisfy registered conservative clearance at3deg; local optimizer not converged, no physical infeasibility claim')
    write(BASE/'revision_registry.json',registry)
    write(BASE/'delivery_check.json',dict(physics_steps_added=0,historical_17_rows_unchanged=True,new_dynamics=0,total_dynamics=17,
        new_real_replay=None,reason='No new actual physics was admitted; offline diagram is not experimental replay',
        ready_reference_exists=(BASE/'reference_ready.npz').exists(),path_gate_exists=(BASE/'path_gate.json').exists(),
        necessary_source_syntax_checked=True,no_SHA=True,old_V3_path='NOT_READY'))
    print(json.dumps(dict(new_dynamics=0,total_dynamics=17,first_partial_collision=first_collision,
        minimum_gap_mm=minimum,length_error_mm=length,rear_nominal_error_mm=rear)),flush=True)


if __name__=='__main__':main()
