"""Continuous local five-segment path from held geometry. Offline only.

Prescribed material lattice, original curve parameters, finite hull constraints
and local SO(3) increments. No radius search or physical dynamics.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import math
import time
import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation,Slerp
from scipy.interpolate import PchipInterpolator
from curve_grasp_revision_geometry import DiscreteFit,CORNER
from curve_guided_geometry import points

BASE=ROOT/'experiments/curve_guided_transfer_20261006/stage_continuation_20261006'
OUT=BASE/'local_path'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class LocalPathFit(DiscreteFit):
    def __init__(self):
        super().__init__()
        with np.load(BASE/'R2_C0/trace.npz') as trace:
            self.held_qpos=trace['qpos'][-1].copy()
            self.held_ctrl=trace['ctrl'][-1].copy()
            self.held_points=trace['points_mm'][-1].copy()
        self.initial_residual=self.held_points-(self.curve.sample(self.curve.lambda0-np.arange(22)*64.5)+CORNER)
        self.previous_desired=None

    def desired(self,feed):
        phase=np.clip(feed/64.5,0,1);fade=1-phase*phase*(3-2*phase)
        return self.curve.sample(self.curve.lambda0+feed-np.arange(22)*64.5)+CORNER+fade*self.initial_residual

    def solve(self,feed,seed):
        desired=self.desired(feed)
        x0=seed.copy()
        if self.previous_desired is not None:
            x0[:3]+=np.mean(desired-self.previous_desired,axis=0)
        Rseed=Rotation.from_rotvec(x0[3:6])
        units=np.r_[np.full(3,10.),np.full(23,.1)]
        def unpack(z):
            x=x0+units*z
            x[3:6]=(Rotation.from_rotvec(units[3:6]*z[3:6])*Rseed).as_rotvec()
            return x
        bounds=[(-3,3)]*3+[(-.8/.1,.8/.1)]*3
        bounds += [((-math.radians(89)-x0[i])/.1,(math.radians(89)-x0[i])/.1) for i in range(6,26)]
        self.last=None;self.set(x0)
        ell=self.curve.material_positions(feed)
        pools=[np.flatnonzero(ell<0),np.flatnonzero(ell>self.curve.bounds[-1])]
        grips=[]
        for side,ids in enumerate(pools):
            if len(ids)<2:raise ValueError('Insufficient nominal material available for two-pipe grasp')
            best=sorted(ids,key=lambda i:abs(self.radial_gap(int(i),side)))[:2]
            grips += [(int(i),side) for i in best]
        calls=[0];last_notice=[time.monotonic()]
        def objective(z):
            x=unpack(z);self.set(x)
            error=(points(self.m,self.d)-desired)/10
            closeness=np.array([self.radial_gap(i,s)-.2 for i,s in grips])/2
            regular=(x[6:]-x0[6:])/math.radians(10)
            return float(np.mean(error**2)+.05*np.mean(closeness**2)+.02*np.mean(regular**2))
        def constraints(z):
            x=unpack(z);self.set(x)
            all_gaps=self.gaps(x)
            near=np.array([self.radial_gap(i,s) for i,s in grips])
            return np.r_[all_gaps-.05,1.-near]
        def progress(z):
            calls[0]+=1
            if time.monotonic()-last_notice[0]>15:
                print(json.dumps(dict(feed_mm=float(feed),iteration=calls[0],physics_steps=0)),flush=True)
                last_notice[0]=time.monotonic()
        result=minimize(objective,np.zeros(26),method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=constraints)],callback=progress,
            options=dict(maxiter=120,ftol=1e-7,eps=1e-5))
        x=unpack(result.x);self.set(x);mujoco.mj_forward(self.m,self.d)
        gaps=self.gaps(x)
        collisions=[dict(geom1=self.m.geom(c.geom1).name,geom2=self.m.geom(c.geom2).name,
                        distance_mm=float(c.dist*1000)) for c in self.d.contact if c.dist<0]
        rms=float(np.sqrt(np.mean(np.sum((points(self.m,self.d)-desired)**2,axis=1))))
        grip_gaps=[float(self.radial_gap(i,s)) for i,s in grips]
        ready=bool(gaps.min()>=.049 and max(grip_gaps)<=1.001 and not collisions and rms<=25)
        row=dict(feed_mm=float(feed),geometric_pose_pass=ready,solver_success=bool(result.success),
            solver_message=str(result.message),iterations=int(result.nit),minimum_conservative_gap_mm=float(gaps.min()),
            grip_modules=[dict(module=i,side='target' if s else 'source',gap_mm=g) for (i,s),g in zip(grips,grip_gaps)],
            desired_rms_mm=rms,maximum_joint_deg=float(np.degrees(abs(x[6:])).max()),
            contacts=collisions,qpos=self.d.qpos.tolist(),actual_points_mm=points(self.m,self.d).tolist())
        self.previous_desired=desired.copy()
        return x,row


def review(fit,feeds,states):
    ip=PchipInterpolator(feeds,states,axis=0)
    rp=Slerp(feeds,Rotation.from_rotvec(states[:,3:6]))
    # Midpoint plus 0.25 mm checks cover this local interval, not arbitrary continuum.
    samples=np.unique(np.r_[np.arange(0,feeds[-1]+1e-8,.25),feeds,(feeds[:-1]+feeds[1:])/2,feeds[-1]])
    rows=[]
    for d in samples:
        x=ip(d);x[3:6]=rp(d).as_rotvec();fit.set(x);mujoco.mj_forward(fit.m,fit.d)
        negative=[dict(geom1=fit.m.geom(c.geom1).name,geom2=fit.m.geom(c.geom2).name,
                      distance_mm=float(c.dist*1000)) for c in fit.d.contact if c.dist<0]
        if negative:rows.append(dict(feed_mm=float(d),contacts=negative))
    q=ip(samples)[:,6:]
    slopes=np.degrees(abs(ip.derivative()(samples)[:,6:])).max(axis=1)
    speeds=np.minimum(5.,9./np.maximum(slopes,1e-9))
    duration=float(np.trapz(1/speeds,samples))
    return dict(geometry_pass=not rows,failures=rows,checked_count=len(samples),spacing_mm=.25,
        maximum_required_speed_at_5mm_s_deg_s=float(slopes.max()*5),
        reparameterized_active_duration_s=duration,
        planned_target_speed_deg_s=9.,retiming_only='speed(d)=min(5 mm/s,9 deg/s / max abs(dq/dd)); validate command sampling separately',
        finite_sampling_limitation='Does not prove collision absence at every arbitrary real feed')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'result.json').exists():raise FileExistsError('Preserve completed local path attempt')
    held=json.loads((BASE/'R2_C0/result.json').read_text(encoding='utf8'))
    if not held['full_holding_pass']:raise RuntimeError('Full holding required before local path work')
    write('plan.json',dict(physics_steps=0,curve_parameters='same R95/P220/rc100/bridge96.75, five BODY segments',
        interval_mm=[0.,161.25],material_lattice_mm=8.0625,
        start='nearest contact-free geometry to actual R2 held shape; any B placement only before first step',
        connector='one offline initial point residual fades over one material link, not online measured-reference reset',
        controller='R2 frozen; executed residual remains indexed by physical joint/body material',
        gap_mm=.05,grip_gap_upper_mm=1.,rms_upper_mm=25.,q_geometry_limit_deg=89.,
        motion_gate='continuous sampled shell collision pass AND valid retiming AND interface holding before B',
        no_body_reset_during_run=True,new_mechanism_round=False))
    fit=LocalPathFit()
    seed=np.r_[fit.held_qpos[:3]*1000,Rotation.from_quat(fit.held_qpos[3:7][[1,2,3,0]]).as_rotvec(),fit.held_qpos[fit.qadr]]
    feeds=np.arange(0,161.25+.001,8.0625)
    states=[];rows=[]
    for feed in feeds:
        seed,row=fit.solve(float(feed),seed)
        states.append(seed.copy());rows.append(row)
        write('progress.json',dict(physics_steps=0,rows=rows))
        np.savez_compressed(OUT/'reference_partial.npz',feed_mm=feeds[:len(states)],states=np.array(states))
        print(json.dumps({k:row[k] for k in ['feed_mm','geometric_pose_pass','iterations','minimum_conservative_gap_mm','desired_rms_mm']},ensure_ascii=False),flush=True)
        if not row['geometric_pose_pass']:break
    outcome=dict(status='NOT_READY',physics_steps=0,rows=rows,controller_revision=2,
        active_transfer_run=False,old_V3_gate_unchanged=True)
    if len(rows)==len(feeds) and all(r['geometric_pose_pass'] for r in rows):
        np.savez_compressed(OUT/'reference_discrete.npz',feed_mm=feeds,states=np.array(states))
        outcome['continuous_review']=review(fit,feeds,np.array(states))
        outcome['status']='CONTINUOUS_REFINEMENT_NEEDED' if not outcome['continuous_review']['geometry_pass'] else 'INTERFACE_AND_TIMING_CHECK_NEEDED'
    write('result.json',outcome)
    print(json.dumps(dict(status=outcome['status'],poses=len(rows),physics_steps=0)),flush=True)


if __name__=='__main__':main()
