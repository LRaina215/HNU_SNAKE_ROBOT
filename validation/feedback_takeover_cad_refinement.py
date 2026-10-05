"""Local CAD-envelope refinement of a right-angle geometry hypothesis.

This is a diagnostic, not an assumption of observed contacts. Near-shell
points with adjacent encoder-error load supply uncertain fitting candidates.
True axes/contact labels are used only after the estimate for scoring.
"""
import numpy as np
from scipy.optimize import least_squares
from segmented_pole_experiment import ROOT,HORIZONTAL,VERTICAL,write_json,pipe_frame
from mujoco_checkpoint import load_checkpoint
from spatial_takeover_experiment import points_for_angles
from encoder_pipe_geometry import right_angle_hypothesis
from encoder_pipe_proximity import EncoderPipeProximity

OUT=ROOT/'experiments/segmented_pole_20261005/feedback_takeover'


def refine_target(query,measured,points,initial,load,target_full_line=True):
    """Fit only the remaining target line once rear control has been released.

    No source constraint, intersection or pipe endpoint is observable from
    one coil. Preserve the fitted pipe radius and solve four line coordinates.
    """
    t=initial['target'];c=initial['corner_mm'];r=initial['pipe_radius_mm']
    h=np.eye(3)[np.argmin(abs(t))];h-=h@t*t;h/=np.linalg.norm(h)
    v=np.cross(t,h)
    gap=query.gaps(measured,points,c,initial['source'],t,r,target_full_line)[1]
    adjacent=np.maximum(np.r_[0,abs(load)],np.r_[abs(load),0])
    selected=(gap<5.)&(adjacent>.1)
    def decode(x):
        target=t+x[2]*h+x[3]*v;target/=np.linalg.norm(target)
        return c+x[0]*h+x[1]*v,target
    def residual(x):
        corner,target=decode(x)
        distances=query.gaps(measured,points,corner,initial['source'],target,r,target_full_line)[1]
        return np.r_[distances[selected]+.5,2*np.minimum(distances+2.,0.),x[:2]/20.,x[2:]/.1]
    fit=least_squares(residual,np.zeros(4),bounds=([-20,-20,-.15,-.15],[20,20,.15,.15]),
        x_scale=[5,5,.05,.05],loss='soft_l1',f_scale=1.,diff_step=1e-3,max_nfev=100)
    corner,target=decode(fit.x)
    return dict(initial,corner_mm=corner,target=target,selected_shell_candidates=selected,
        parameter_change=fit.x,cost=float(fit.cost),solver_success=bool(fit.success),
        target_only=True)


def refine(query,measured,points,initial,load,fixed_radius=False,candidate_age=None,period=.2):
    s=initial['source'];t=initial['target'];c=initial['corner_mm'];r=initial['pipe_radius_mm']
    v=np.cross(s,t)
    gap=query.gaps(measured,points,c,s,t,r)
    adjacent=np.maximum(np.r_[0,abs(load)],np.r_[abs(load),0])
    selected=(gap<5.)&(adjacent[None,:]>.1)
    fresh=selected.copy();weights=np.ones_like(gap)
    if candidate_age is not None:
        candidate_age=np.asarray(candidate_age)+period
        candidate_age[fresh]=0.
        retained=(candidate_age<=.6+1e-9)&(adjacent[None,:]>.1)
        selected|=retained
        weights=np.exp(-candidate_age/.3)
    def decode(x):
        source=s+x[3]*t+x[4]*v;source/=np.linalg.norm(source)
        normal=t-(t@source)*source;normal/=np.linalg.norm(normal)
        target=np.cos(x[5])*normal+np.sin(x[5])*np.cross(source,normal)
        return c+x[:3],source,target,r+(0. if fixed_radius else x[6])
    def residual(x):
        corner,source,target,radius=decode(x)
        distances=query.gaps(measured,points,corner,source,target,radius)
        # A soft 0.5mm contact deformation is a model prior, not a label.
        return np.r_[(distances[selected]+.5)*np.sqrt(weights[selected]),
            2*np.minimum(distances.reshape(-1)+2.,0.),x[:3]/20.,x[3:6]/.1,
            0. if fixed_radius else x[6]/5.]
    count=6 if fixed_radius else 7
    fitted=least_squares(residual,np.zeros(count),
        bounds=([-20,-20,-20,-.15,-.15,-.2,-8][:count],[20,20,20,.15,.15,.2,8][:count]),
        x_scale=[5,5,5,.05,.05,.05,3][:count],loss='soft_l1',f_scale=1.,
        diff_step=1e-3,max_nfev=100)
    corner,source,target,radius=decode(fitted.x)
    return dict(corner_mm=corner,source=source,target=target,pipe_radius_mm=radius,
        geometric_candidate=initial['geometric_candidate'],selected_shell_candidates=selected,
        initial_gap_mm=gap,refined_gap_mm=query.gaps(measured,points,corner,source,target,radius),
        parameter_change=fitted.x,cost=float(fitted.cost),solver_success=bool(fitted.success),
        fresh_shell_candidates=fresh,candidate_age=candidate_age)


def run():
    rows=[]
    for scene,cp in [('horizontal',HORIZONTAL),('vertical_down',VERTICAL)]:
        m,d,meta,_=load_checkpoint(cp,verify_hashes=False)
        measured=d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]]
        points=points_for_angles(m,d,measured)
        initial=right_angle_hypothesis(points,meta['settings']['radius'])
        load=np.clip(m.actuator_gainprm[:,0]*(d.ctrl-measured),m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        fitted=refine(EncoderPipeProximity(m),measured,points,initial,load)
        corner,source,target,r=pipe_frame(m)
        scores=dict(source_axis_error_deg=float(np.degrees(np.arccos(np.clip(fitted['source']@source,-1,1)))),
            target_axis_error_deg=float(np.degrees(np.arccos(np.clip(fitted['target']@target,-1,1)))),
            corner_error_mm=float(np.linalg.norm(fitted['corner_mm']-corner)),radius_error_mm=float(fitted['pipe_radius_mm']-r))
        row=dict(scene=scene,estimate=fitted,scoring=scores);rows.append(row);print(scene,scores,flush=True)
    write_json(OUT/'cad_geometry_refinement.json',dict(rows=rows,
        limitations='Near-shell/load constraints are uncertain. Two selected captures; no contact or axis truth used by solver.'))


if __name__=='__main__':run()
