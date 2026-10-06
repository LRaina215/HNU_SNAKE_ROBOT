"""Reference boundary consistency repair, frozen R2 mechanism, no mj_step."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from curve_local_path import LocalPathFit,BASE
from curve_local_path_refine import inspect,repair
from curve_guided_geometry import points,CORNER
from snake_control.curve_grasp_revision import FittedMaterialCurve
from snake_control.local_material_reference import BoundaryMaterialCurve

OUT=BASE/'local_path_boundary'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class BoundaryFit(LocalPathFit):
    def __init__(self):
        super().__init__()
        self.old=FittedMaterialCurve(BASE.parent/'structural_revision_v3/fitted_reference.npz')
        self.initial=self.old.state(0.)
        self.start_tangent=self.old.feed_tangent(0.)
        self.set(self.initial)
        self.initial_residual=points(self.m,self.d)-(self.curve.sample(self.curve.lambda0-np.arange(22)*64.5)+CORNER)

    def seed_next(self,feed,seed,previous_feed):
        # Same desired material lattice and original guide, no fresh initial
        # pose search. A prior successful local pose is only a warm start.
        x=seed.copy();x[:3]+=np.mean(self.desired(feed)-self.desired(previous_feed),axis=0)
        return x


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'sources').mkdir()
    for name in ['validation/curve_boundary_path.py','validation/curve_local_path.py',
                 'validation/curve_local_path_refine.py','snake_control/local_material_reference.py']:
        shutil.copy2(ROOT/name,OUT/'sources'/Path(name).name)
    fit=BoundaryFit()
    write('plan.json',dict(physics_steps=0,mechanism_revision=2,new_control_mechanism=False,
        issue='interface0 replaced old nominal, tangent and prior while keeping only acknowledged output; restoring equilibrium changed',
        correction='fixed d0 saved V3 nominal and dq/dd; inherit old R2 initial prior, acknowledged residual and controller age',
        unchanged='R2 executable methods, .5s restoring, K .01/.02, physics, constraints, material body family',
        curve_parameters='same five BODY segments, R95/P220/arc100/bridge96.75',
        physical_initial='same actual saved R2 final qpos/ctrl; qvel zero only before first step',
        reference_zero='exact saved V3 nominal world/root/joints, not a new measured nominal zero',
        geometric_nominal_limitation='nominal reference shells only; retained grasp deformation/contact and actual free-body motion require dynamics',
        gap_mm=.05,rms_upper_mm=25.,q_geometry_limit_deg=89.,old_V3_full_path='NOT_READY'))
    feeds=np.arange(0,161.25+.001,8.0625)
    states=[fit.initial.copy()];rows=[];seed=fit.initial.copy()
    # Fixed saved first pose: no fit at d0. Its independently checked geometry
    # is retained, and every new connection is checked below.
    for feed in feeds[1:]:
        guess=fit.seed_next(float(feed),seed,float(feeds[len(states)-1]))
        seed,row=repair(fit,float(feed),guess)
        states.append(seed.copy());rows.append(row)
        write('progress.json',dict(physics_steps=0,rows=rows))
        np.savez_compressed(OUT/'reference_partial.npz',feed_mm=feeds[:len(states)],states=np.array(states),
            start_tangent_rad_per_mm=fit.start_tangent)
        print(json.dumps(row),flush=True)
        if not row['pose_pass']:
            write('refinement_result.json',dict(status='NOT_READY',physics_steps=0,failed_pose=row));return
    states=np.array(states)
    for pass_index in range(6):
        archive=OUT/f'refinement_{pass_index:02d}.npz'
        np.savez_compressed(archive,feed_mm=feeds,states=states,start_tangent_rad_per_mm=fit.start_tangent)
        reference=BoundaryMaterialCurve(archive);review=inspect(fit,reference)
        write(f'refinement_{pass_index:02d}_review.json',review)
        print(json.dumps({k:v for k,v in review.items() if k!='failures'}),flush=True)
        if review['geometry_pass']:
            shutil.copy2(archive,OUT/'reference_ready.npz')
            write('refinement_result.json',dict(status='INTERFACE_AND_TIMING_CHECK_NEEDED',physics_steps=0,
                final_pass=pass_index,knots=len(feeds),review=review,boundary_nominal_identity_error_rad=0.,
                boundary_tangent_identity_error_rad=0.,controller_revision=2,old_V3_gate_unchanged=True));return
        if pass_index==5:
            write('refinement_result.json',dict(status='NOT_READY',physics_steps=0,review=review,
                stop='remaining geometric connection failures after fixed-boundary collocation'));return
        indices=sorted(set(min(np.searchsorted(feeds,r['feed_mm'],side='right')-1,len(feeds)-2) for r in review['failures']))
        additions=[];insertions=[]
        for index in indices:
            midpoint=float((feeds[index]+feeds[index+1])/2)
            x,row=repair(fit,midpoint,reference.state(midpoint))
            additions.append((midpoint,x));insertions.append(row)
            write(f'refinement_{pass_index:02d}_insertions.json',dict(physics_steps=0,rows=insertions))
            print(json.dumps(dict(pass_index=pass_index,**row)),flush=True)
            if not row['pose_pass']:
                write('refinement_result.json',dict(status='NOT_READY',physics_steps=0,failed_insertion=row));return
        merged=sorted(list(zip(feeds,states))+additions,key=lambda v:v[0])
        feeds=np.array([d for d,x in merged]);states=np.array([x for d,x in merged])


if __name__=='__main__':main()
