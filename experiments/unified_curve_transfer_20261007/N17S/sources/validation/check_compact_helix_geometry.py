"""Necessary saved-CLOSE geometric checks; never advance actual dynamics."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.hook_pull_review_support import restore_saved_state
from snake_control.compact_helix_transfer import CompactHelixTransfer
from snake_control.axis_wrap_transfer import AxisWrapTransfer
from snake_control.unified_curve_transfer import LINK,BODY_RADIUS


def check(destination):
    destination.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m)
    restore_saved_state(m,d,ROOT/'experiments/hook_pull_20261006/CLOSE1B/final_state.npz')
    new=CompactHelixTransfer(m,d);old=AxisWrapTransfer(m,d)
    new.retain_observed_grip_clearance(d,0.,None)
    shifts=np.unique(np.r_[np.arange(0.,.251,.005),np.arange(.26,1.701,.01)])
    orientation=new.root_orientation.copy();rows=[]
    for shift in shifts:
        guide,_,locations=new.curve.reference(float(shift))
        axis=guide[0]-guide[1];axis/=np.linalg.norm(axis)
        cross=np.cross(orientation[:,2],axis);sine=np.linalg.norm(cross)
        orientation=Rotation.from_rotvec(cross*math.atan2(sine,float(orientation[:,2]@axis))/max(sine,1e-12)).as_matrix()@orientation
        angles,_,_=new.integral_targets(guide,orientation,np.zeros(2))
        rows.append(dict(shift_mm=float(shift*1000),maximum_raw_integral_angle_deg=float(np.degrees(abs(angles)).max()),
            minimum_guide_solid_clearance_mm=float(new.pipe_geometry.link_minimum(guide,65).min()*1000),
            maximum_guide_chord_error_mm=float(abs(np.linalg.norm(np.diff(guide,axis=0),axis=1)-LINK).max()*1000),
            all_nodes_beyond_compact_transition=bool(locations.min()>new.curve.node_s0[0]+3*LINK)))
    a=new.compact_curve_action;radius=a['target_radius_mm']/1000;b=a['new_pitch_mm']/1000/(2*math.pi)
    distance=lambda theta:math.sqrt(2*radius*radius*(1-math.cos(theta))+(b*theta)**2)
    adjacent_turn=minimize_scalar(distance,bounds=(1.5*math.pi,2*math.pi),method='bounded')
    # Account for the inward bow of a 64.5 mm straight link as a conservative
    # allowance; ideal turn distance alone is not complete shell clearance.
    angular_step=LINK*math.sqrt(1-a['new_axial_fraction']**2)/radius
    sag=radius*(1-math.cos(.5*angular_step))
    head=float(new.curve.node_s0[0]);point_error=float(np.linalg.norm(new.curve.base(head)-old.curve.base(head)))
    second_derivative_jump=float(np.linalg.norm(new.curve.base(head+1e-9,2)-old.curve.base(head-1e-9,2)))
    report=dict(curve_comparison=a,checks=dict(number_of_common_windows=len(rows),shift_range_mm=[0.,1700.],
        initial_curve_exactly_preserved=new.initial_fixed_route_error<1e-9,
        join_position_error_mm=point_error*1000,join_first_derivative_error=a['join_first_derivative_error'],
        numerical_join_second_derivative_jump_per_m=second_derivative_jump,
        maximum_raw_integral_angle_deg=max(r['maximum_raw_integral_angle_deg'] for r in rows),
        minimum_nominal_centerline_original_solid_gap_mm=min(r['minimum_guide_solid_clearance_mm'] for r in rows),
        ideal_next_turn_minimum_centerline_distance_mm=adjacent_turn.fun*1000,
        conservative_uniform_turn_shell_gap_mm=(adjacent_turn.fun-2*sag-2*BODY_RADIUS)*1000,
        no_actual_dynamics_advanced=True,
        scope='initial material phase transported continuously; guide/angle feasibility and ideal turn shell separation, not proof of contact-constrained realizability'),windows=rows)
    (destination/'preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    u=np.linspace(head-.45,head+1.7,1600);o=old.curve.base(u);n=new.curve.base(u)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for p,label in [(o,'N15: pitch 314.6 mm'),(n,'Compact: pitch 180 mm')]:
        phi=np.unwrap(np.arctan2(-(p[:,2]-1),p[:,0]-1.1))
        axs[0].plot((u-head)*1000,p[:,1]*1000,label=label)
        axs[1].plot(p[:,1]*1000,phi,label=label)
    axs[0].axvline(0,color='gray',linestyle='--');axs[0].axvline(3*LINK*1000,color='gray',linestyle=':')
    axs[0].set_xlabel('Curve parameter beyond initial head / mm');axs[0].set_ylabel('Target axis Y / mm')
    axs[1].set_xlabel('Target axis Y / mm');axs[1].set_ylabel('Inherited polar angle / rad')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.savefig(destination/'curve_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(report['checks']))


if __name__=='__main__':check(ROOT/'experiments/unified_curve_transfer_20261007/COMPACT_HELIX_20261009')
