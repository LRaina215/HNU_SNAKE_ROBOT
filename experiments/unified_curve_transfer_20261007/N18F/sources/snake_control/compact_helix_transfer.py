"""One fixed CLOSE route, smoothly continued to a compact target helix."""
import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline,PPoly
from scipy.integrate import cumulative_trapezoid
from snake_control.axis_wrap_transfer import AxisWrapTransfer
from snake_control.global_transfer_route import FrozenRouteSpec
from snake_control.unified_curve_transfer import LINK,BODY_RADIUS


class CompactHelixTransfer(AxisWrapTransfer):
    def __init__(self,model,data,pitch=.180,skin_aware=True):
        super().__init__(model,data)
        self.skin_aware=skin_aware;self.skin_feedback={}
        c=self.curve;old=c.base;head=float(c.node_s0[0])
        radius=float(c.radii[1]);alpha=pitch/math.sqrt(pitch*pitch+(2*math.pi*radius)**2)
        direction=math.copysign(1.,c.theta_rates[1]);rate=direction*math.sqrt(1-alpha*alpha)/radius
        transition=3*LINK
        extra=np.unique(np.r_[np.linspace(head,c.domain[1],math.ceil((c.domain[1]-head)/.005)+1),head+transition])
        native=old(extra);derivative=old(extra,1)
        radial=native[:,[0,2]]-[1.1,1.];rr=np.linalg.norm(radial,axis=1)
        angular_rate=(radial[:,0]*derivative[:,2]-radial[:,1]*derivative[:,0])/rr**2
        z=np.clip((extra-head)/transition,0.,1.);blend=z**3*(10-15*z+6*z*z)
        blended_rate=(1-blend)*angular_rate+blend*rate
        blended_axis=(1-blend)*derivative[:,1]+blend*alpha
        angle0=math.atan2(radial[0,1],radial[0,0])
        angle=angle0+cumulative_trapezoid(blended_rate,extra,initial=0.)
        axis=native[0,1]+cumulative_trapezoid(blended_axis,extra,initial=0.)
        r=(1-blend)*rr+blend*radius
        extension=np.column_stack([1.1+r*np.cos(angle),axis,1.+r*np.sin(angle)])
        end_derivative=np.array([-radius*rate*np.sin(angle[-1]),alpha,radius*rate*np.cos(angle[-1])])
        future=CubicSpline(extra,extension,axis=0,bc_type=((1,old(head,1)),(1,end_derivative)))
        cut=int(np.searchsorted(old.x,head))
        if abs(float(old.x[cut])-head)>1e-10:raise ValueError('Head continuation must be an existing CLOSE spline knot')
        c.base=PPoly(np.concatenate([old.c[:,:cut],future.c],axis=1),np.r_[old.x[:cut],future.x])
        c.arc_grid_u=np.unique(np.r_[c.arc_grid_u,extra])
        old_alpha=float(c.axial_sines[1]);old_pitch=2*math.pi*radius*old_alpha/math.sqrt(1-old_alpha*old_alpha)
        c.axial_sines[1]=alpha;c.theta_rates[1]=rate
        spec=dict(c.planned_route_spec)
        spec.update(version=6,origin='CLOSE fixed spatial curve, same handed tangent continuation to one compact target helix',
            spline_x=c.base.x.tolist(),spline_c=c.base.c.tolist(),
            pipe_axial_sines=list(c.axial_sines),pipe_polar_rates=list(c.theta_rates),
            target_helix_pitch_m=pitch,target_compact_transition_start_parameter=head,
            target_compact_transition_length_m=transition)
        c.planned_route_spec=FrozenRouteSpec(spec)
        initial,_,_=c.reference(0.);actual,_=self.observe(data)
        self.initial_fixed_route_error=float(np.linalg.norm(initial-actual,axis=1).max())
        if self.initial_fixed_route_error>1e-9:raise ValueError('Compact future changed initial CLOSE body')
        tangent_error=float(np.linalg.norm(c.base(head,1)-old(head,1)))
        self.compact_curve_action=dict(action='fixed route continues current hook tangent and handedness; future axial pitch decreases smoothly over three rods, then stays 180 mm',
            physical_body_length_mm=21*LINK*1000,physical_shell_diameter_mm=2*BODY_RADIUS*1000,
            target_radius_mm=radius*1000,old_pitch_mm=old_pitch*1000,new_pitch_mm=pitch*1000,
            old_axial_fraction=old_alpha,new_axial_fraction=alpha,
            ideal_all_body_old_axis_span_mm=21*LINK*old_alpha*1000,ideal_all_body_new_axis_span_mm=21*LINK*alpha*1000,
            ideal_all_body_old_turns=21*LINK*math.sqrt(1-old_alpha*old_alpha)/(2*math.pi*radius),
            ideal_all_body_new_turns=21*LINK*math.sqrt(1-alpha*alpha)/(2*math.pi*radius),
            axial_span_reduction_fraction=1-alpha/old_alpha,
            ideal_new_turn_normal_shell_gap_mm=(pitch*math.sqrt(1-alpha*alpha)-2*BODY_RADIUS)*1000,
            new_helix_curvature_per_m=(1-alpha*alpha)/radius,new_helix_torsion_per_m=direction*alpha*math.sqrt(1-alpha*alpha)/radius,
            future_transition_length_mm=transition*1000,initial_route_fit_error_mm=self.initial_fixed_route_error*1000,
            join_first_derivative_error=tangent_error,initial_material_normals='same CLOSE calibration as N15; retained without actual-q refit',
            compactness_caveat='ideal uniform-helix totals exclude inherited hook and transition; actual wrapping requires dynamics')

    def pipe_skin_penetration(self,data):
        penetration=np.zeros(21)
        for contact in data.contact:
            for robot,pipe in [(contact.geom1,contact.geom2),(contact.geom2,contact.geom1)]:
                body=int(self.m.geom_bodyid[robot])
                if self.head<=body<self.head+21 and (self.m.geom(pipe).name or '').startswith('pipe_'):
                    penetration[body-self.head]=max(penetration[body-self.head],-float(contact.dist))
        return penetration

    def evaluate(self,parameters,shift,anchor,head_u,data,patch=None):
        value=super().evaluate(parameters,shift,anchor,head_u,data,patch)
        if not self.skin_aware:return value
        actual=self.pipe_skin_penetration(data)
        registration=self.prediction_registration
        rotation=np.asarray(registration.get('rotation',np.eye(3)));translation=np.asarray(registration.get('translation_mm',np.zeros(3)))/1000
        d=self.fit_data
        actual_rotation=data.xmat[self.head].reshape(3,3)
        d.qpos[:3]=rotation@data.qpos[:3]+translation
        d.qpos[3:7]=Rotation.from_matrix(rotation@actual_rotation).as_quat()[[3,0,1,2]]
        mujoco.mj_fwdPosition(self.m,d)
        predicted=self.pipe_skin_penetration(d)
        allowance=np.clip(actual+.0005,.00075,.002)
        penalty=4.*np.maximum(0.,predicted-allowance)
        self.skin_feedback=dict(action='original collision-shell depth augments four-parameter curve residual; no joint-angle patch or physical root assignment',
            predicted_full_goal_pipe_skin_penetration_mm=(predicted*1000).tolist(),actual_pipe_skin_penetration_mm=(actual*1000).tolist(),
            predicted_skin_allowance_mm=(allowance*1000).tolist(),skin_penalty_multiplier=4.,
            maximum_predicted_goal_pipe_skin_penetration_mm=float(predicted.max()*1000),
            maximum_actual_pipe_skin_penetration_mm=float(actual.max()*1000),
            warning='bearing-pose registered full future target, not next ctrl; soft feedback allows progress and is not a proof of contact feasibility')
        return (*value,penalty)

    def additional_prediction_residual(self,value,data):
        return value[8] if self.skin_aware else np.zeros(0)

    def command(self,data,t,forces,torque,support_forces=None):
        target,info=super().command(data,t,forces,torque,support_forces)
        info['stage']='compact_target_helix_with_bearing_located_feedback'
        info['compact_helix']=self.compact_curve_action.copy()
        if self.skin_aware:info['skin_curve_feedback']=self.skin_feedback.copy()
        return target,info
