"""One M8 approach/handoff candidate and at most one truth-assisted trial.

Private kinematics: retain four support material points, let M7/M14 slide,
curl M8 toward the target tube and guide M9 outside the junction. Only after
actual M8 support is observed may the original M7/M14 support give way.
No physical root, qpos or qvel updates after the saved-state restore.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]
import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
import hook_pull_step4_rolling as kin
import hook_pull_short_physics as record
from snake_control.hook_pull_entry_prefix import FixedPreloadEntry
from snake_control.hook_pull_review_support import restore_saved_state

OUT = ROOT / "experiments/hook_pull_review_20261007/m8_transfer"
APPROACH, HANDOFF, DT = 14., 4., .05
ACTION_MAX, HOLD, WAIT_MAX = 20., 10., 2.
PINNED = [(0, "target"), (3, "target"), (17, "source"), (20, "source")]
NORMAL = [(7, "target"), (14, "source")]
PIPE_RADIUS = .0565
write = record.write


def smooth(u):
    u = np.clip(u, 0., 1.)
    return float(u*u*(3.-2.*u))


def skew(x):
    return np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])


class Candidate:
    """One fixed task definition, with no parameter search or physical steps."""
    def __init__(self):
        self.model = kin.Model()
        m = self.model
        restore_saved_state(m.m, m.d, kin.START)
        m.update()
        self.q0, self.held = m.d.qpos.copy(), m.d.ctrl.copy()
        self.shape0 = self.q0[m.qadr].copy()
        self.entry = FixedPreloadEntry(self.held, self.shape0)
        self.p0 = m.nodes().copy()
        self.c0 = (self.p0[:-1]+self.p0[1:])/2
        self.u0 = (self.p0[9]-self.p0[8])/np.linalg.norm(self.p0[9]-self.p0[8])
        _, n = kin.radial(self.c0[8], kin.PIPES["target"][0]+self.c0[8,1]*np.array([0.,1.,0.]))
        # Extend the hook tangentially around the tube, not outward/upward.
        self.c8 = kin.PIPES["target"][0] + .0848*n
        self.c8[1] = self.c0[8,1]+.030
        self.u8 = .8*np.array([-n[2],0.,n[0]])+np.array([0.,-.6,0.])
        # M9 goes outside the source tube; this end point is lower than its start.
        u9 = np.array([-.45,-.55,np.sqrt(1.-.45**2-.55**2)])
        self.c9 = self.c8+.03225*(self.u8+u9)
        self.anchors, self.rho = {}, {}
        for k, pipe in PINNED:
            candidates = []
            for c in m.d.contact:
                for g, b in [(c.geom1,int(m.m.geom_bodyid[c.geom2])),
                             (c.geom2,int(m.m.geom_bodyid[c.geom1]))]:
                    if b == m.head+k and g in m.pipe_geoms[pipe]:
                        candidates.append((float(c.dist), c.pos.copy()))
            if not candidates:
                raise ValueError("Missing initial support contact: " + str((k,pipe)))
            world = min(candidates, key=lambda x:x[0])[1]
            frame = m.d.xmat[m.head+k].reshape(3,3)
            self.anchors[(k,pipe)] = (world, frame.T@(world-m.d.xpos[m.head+k]))
        for k,pipe in NORMAL:
            x,foot = kin.closest(self.p0[k],self.p0[k+1],pipe)
            self.rho[(k,pipe)] = kin.radial(x,foot)[0]
        self.floor = {}
        for k in range(21):
            for pipe in kin.PIPES:
                x,foot=kin.closest(self.p0[k],self.p0[k+1],pipe)
                self.floor[(k,pipe)] = min(.0845,kin.radial(x,foot)[0])-.0005
        offset=self.held-self.shape0
        self.low=np.maximum(-kin.LIMIT_GATE,-kin.LIMIT_GATE-offset)
        self.high=np.minimum(kin.LIMIT_GATE,kin.LIMIT_GATE-offset)
        self.scale=np.r_[np.full(3,.01),np.full(m.m.nv-3,np.radians(10.))]

    def desired(self,t):
        s=smooth(t/APPROACH); h=smooth((t-APPROACH)/HANDOFF)
        c8=self.c0[8]+s*(self.c8-self.c0[8]); c8[1]+=.005*h
        c9=self.c0[9]+s*(self.c9-self.c0[9]); c9[1]+=.005*h
        axis=self.u0+s*(self.u8-self.u0);axis/=np.linalg.norm(axis)
        return dict(c8=c8,c9=c9,axis=axis,rear_x=self.c0[13,0]+.006*s+.004*h,
                    release=.0015*h)

    def velocity(self,t,dt,previous):
        m=self.model; nodes=m.nodes(); centers=(nodes[:-1]+nodes[1:])/2
        a,b=self.desired(t),self.desired(t+dt)
        E,e,P,p,G,g=[],[],[],[],[],[]
        for (k,pipe),(want,local) in self.anchors.items():
            frame=m.d.xmat[m.head+k].reshape(3,3)
            point=m.d.xpos[m.head+k]+frame@local
            E.extend(1000*m.jac(point,k)[0]);e.extend(3000*(want-point))
        for k,pipe in NORMAL:
            x,foot=kin.closest(nodes[k],nodes[k+1],pipe);r,n=kin.radial(x,foot)
            E.append(1000*n@m.jac(x,k)[0])
            e.append(1000*((b['release']-a['release'])/dt+
                          3.*(self.rho[(k,pipe)]+a['release']-r)))
        for k,key,weight in [(8,'c8',4.),(9,'c9',1.)]:
            J=m.jac(centers[k],k)[0]
            P.extend(weight*1000*J)
            p.extend(weight*1000*((b[key]-a[key])/dt+3.*(a[key]-centers[k])))
        axis=(nodes[9]-nodes[8])/np.linalg.norm(nodes[9]-nodes[8])
        Ja=-skew(axis)@m.jac(centers[8],8)[1]
        P.extend(40.*Ja)
        p.extend(40.*((b['axis']-a['axis'])/dt+3.*(a['axis']-axis)))
        P.append(.7*1000*m.jac(centers[13],13)[0][0])
        p.append(.7*1000*((b['rear_x']-a['rear_x'])/dt+3.*(a['rear_x']-centers[13,0])))
        pinned=set(PINNED)
        for k in range(21):
            for pipe in kin.PIPES:
                if (k,pipe) in pinned:continue
                x,foot=kin.closest(nodes[k],nodes[k+1],pipe)
                if kin.MITER[pipe](foot):continue
                radius,n=kin.radial(x,foot);floor=self.floor[(k,pipe)]
                if radius < floor+.012:
                    G.append(1000*n@m.jac(x,k)[0]);g.append(1000*(floor-radius)/dt)
        for i in range(21):
            for j in range(i+2,21):
                v=centers[i]-centers[j];distance=np.linalg.norm(v)
                if distance<.075:
                    G.append(1000*(v/distance)@(m.jac(centers[i],i)[0]-m.jac(centers[j],j)[0]))
                    g.append(1000*(.058-distance)/dt)
        # All rows use mm/s (orientation scaled by 40 mm/rad).
        E=np.asarray(E)*self.scale;e=np.asarray(e)
        P=np.asarray(P)*self.scale;p=np.asarray(p)
        G=np.asarray(G).reshape(-1,m.m.nv)*self.scale;g=np.asarray(g)
        reg=np.r_[np.full(3,100.),np.full(3,2.),np.full(20,2.)]*self.scale
        H=P.T@P+np.diag(reg**2);rhs=P.T@p
        low=np.full(m.m.nv,-1.);high=-low
        low[:3],high[:3]=-5.,5.;low[3:6],high[3:6]=-2.8,2.8
        q=m.d.qpos[m.qadr]
        low[m.vadr]=np.maximum(-1.,(self.low-q)/dt/self.scale[m.vadr])
        high[m.vadr]=np.minimum(1.,(self.high-q)/dt/self.scale[m.vadr])
        constraints=[dict(type='eq',fun=lambda z:E@z-e,jac=lambda z:E)]
        if len(G):constraints.append(dict(type='ineq',fun=lambda z:G@z-g,jac=lambda z:G))
        start=np.minimum(high,np.maximum(low,previous/self.scale))
        answer=minimize(lambda z:.5*z@H@z-rhs@z,start,jac=lambda z:H@z-rhs,
            method='SLSQP',bounds=list(zip(low,high)),constraints=constraints,
            options=dict(ftol=1e-9,maxiter=160))
        if not answer.success or abs(E@answer.x-e).max()>1e-5 or (len(G) and (G@answer.x-g).min() < -1e-5):
            raise RuntimeError('Constrained approach blocked at %.2fs: %s; M8 error %.3f mm'%
                (t,answer.message,1000*np.linalg.norm(a['c8']-centers[8])))
        return answer.x*self.scale


def module_metrics(m,d,k,head):
    n=record.points(m,d);c=(n[k]+n[k+1])/2
    body=head+k;robot_geoms=np.flatnonzero(m.geom_bodyid==body)
    distances=[]
    for names in [['pipe_straight_01','pipe_sharp_in_02'],['pipe_sharp_out_03','pipe_straight_04']]:
        distances.append(min(mujoco.mj_geomDistance(m,d,int(g),m.geom(name).id,1.,None)*1000
                             for g in robot_geoms for name in names))
    return np.r_[c,np.linalg.norm(c[[0,2]]-[1100.,1000.]),
                 np.linalg.norm(c[[1,2]]-[0.,1000.]),distances,(n[k+1]-n[k])/64.5]


def plan():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'plan_started.json').exists():raise FileExistsError('Sole candidate already attempted; no refit')
    write(OUT/'registration.json',dict(date='2026-10-07',candidate='M8_inward_curl_then_local_handoff',
        start=str(kin.START.relative_to(ROOT)),previous_dynamics_trials=2,total_limit=4,this_dispatch_max_trials=1,
        approach_s=APPROACH,handoff_s=HANDOFF,waiting_max_s=WAIT_MAX,action_max_s=ACTION_MAX,post_action_hold_s=HOLD,
        pinned_material_points=PINNED,sliding_normal_support=NORMAL,M8_center_radius_m=.0848,
        M8_axial_approach_mm=30.,M8_handoff_feed_mm=5.,rear_M13_feed_mm=[6.,4.],
        M7_M14_local_radial_relief_mm=1.5,
        control='fixed-preload shape increments; actual M8 force/region gates handoff (simulation truth assisted)',
        gate='M8 y>=75mm, target axis distance<=92mm, target force>0.1N for 0.2s; wait<=2s',
        limits=dict(joint_deg=88.,command_rate_deg_s=10.,pipe_penetration_mm=3.,self_penetration_mm=.5,wrap_min_deg=180.),
        danger='reuse short_physics.Trial.danger every10ms; additionally stop M8 loss lasting0.2s after handoff',
        scoring='M8 actual migration and continuous region/contact from handoff through all10s; final10s mean M8 upward target force>0.1N and M7 upward force or M14 normal force decreases>0.1N relative to last0.5s before handoff; original final joint-support holding gate',
        forbidden=['root/qpos/qvel driving after start','new C0','friction/torque edits','second candidate or dynamics retry']))
    write(OUT/'plan_started.json',dict(sole_candidate=True,budget_s=1200,physics_steps=0))
    started=time.monotonic();candidate=None;times=[0.];poses=[];controls=[];failure=None
    try:
        candidate=Candidate();m=candidate.model;poses=[m.d.qpos.copy()];controls=[candidate.held.copy()]
        write(OUT/'task_targets.json',dict(initial_M8_center_mm=candidate.c0[8]*1000,
            approach_M8_center_mm=candidate.c8*1000,approach_M8_axis=candidate.u8,
            approach_M9_center_mm=candidate.c9*1000))
        velocity=np.zeros(m.m.nv)
        for tick in range(round((APPROACH+HANDOFF)/DT)):
            if time.monotonic()-started>1200:raise TimeoutError('Single candidate20min budget')
            velocity=candidate.velocity(tick*DT,DT,velocity)
            mujoco.mj_integratePos(m.m,m.d.qpos,velocity,DT);m.update()
            times.append((tick+1)*DT);poses.append(m.d.qpos.copy());controls.append(candidate.entry(m.d.qpos[m.qadr]))
            if tick%40==39:print(json.dumps(dict(planned_s=times[-1],M8_center_mm=(1000*((m.nodes()[8]+m.nodes()[9])/2)).tolist())),flush=True)
    except Exception as exc:failure=repr(exc)
    times,poses,controls=np.asarray(times),np.asarray(poses),np.asarray(controls)
    np.savez_compressed(OUT/'planned_path.npz',time_s=times,predicted_qpos=poses,ctrl_rad=controls)
    geometry=[];first_bad=None;max_depth=0.;minimum_wrap=360.;target_end=False
    if candidate is not None and len(poses):
        m=candidate.model
        # One full path check, including four intermediate configurations per50ms step.
        for i,q in enumerate(poses):
            checks=[(times[i],q)]
            if i+1<len(poses):
                v=np.empty(m.m.nv);mujoco.mj_differentiatePos(m.m,v,DT,q,poses[i+1])
                for h in (.01,.02,.03,.04):
                    x=q.copy();mujoco.mj_integratePos(m.m,x,v,h);checks.append((times[i]+h,x))
            for t,x in checks:
                m.d.qpos[:]=x;m.update();pairs,depth,ground,selfhit=m.touching()
                wrap=kin.wrap_deg(m.nodes());max_depth=max(max_depth,-1000*depth);minimum_wrap=min(minimum_wrap,wrap)
                mm=module_metrics(m.m,m.d,8,m.head);mm9=module_metrics(m.m,m.d,9,m.head)
                geometry.append(np.r_[t,mm,mm9,-1000*depth,wrap])
                bad=[]
                if np.degrees(abs(x[m.qadr])).max()>88.+1e-9:bad.append('predicted_joint_above88')
                if depth<-.003:bad.append('finite_pipe_penetration_above3mm')
                if ground or selfhit:bad.append('ground_or_finite_nonadjacent_self_penetration')
                if wrap<180.:bad.append('wrap_below180')
                # Potential support, not a dynamic force assertion.
                for pipe,count in [('source',2),('target',1)]:
                    near={k for (k,p),d in pairs.items() if p==pipe and d<.001}
                    if len(near)<count:bad.append(pipe+'_geometric_support_discontinued')
                if bad and first_bad is None:first_bad=dict(time_s=float(t),failures=bad)
        final=geometry[-1];target_end=bool(final[2]>=75. and final[4]<=92. and final[7]<=.3)
        if not target_end and failure is None:failure='M8_final_finite_shell_not_in_target_contact_region'
        if first_bad is not None and failure is None:failure=str(first_bad)
        if len(controls)>1:
            rate=float(np.degrees(abs(np.diff(controls,axis=0))).max()/DT)
            if rate>10.+1e-8 or np.degrees(abs(controls)).max()>88.+1e-9:
                failure=failure or 'command_rate_or_absolute_limit'
        else:rate=0.
        np.savez_compressed(OUT/'geometry_trace.npz',rows=np.array(geometry),
            columns='t;M8 center_xyz,target_axis_radius,source_axis_radius,source_surface_gap,target_surface_gap,axis_xyz;M9 same;penetration;wrap')
    else:rate=0.
    result=dict(ready=failure is None and len(times)==361,stop_reason=failure,first_geometry_failure=first_bad,
        completed_path_s=float(times[-1]),physics_steps=0,sole_candidate=True,wall_s=time.monotonic()-started,
        maximum_command_rate_deg_s=rate,maximum_command_angle_deg=float(np.degrees(abs(controls)).max()) if controls.size else None,
        maximum_pipe_penetration_mm=max_depth,minimum_wrap_deg=minimum_wrap,M8_final_region=target_end,
        limits='private support/contact geometry only; no friction/load/dynamic guarantee; predicted root never actuated')
    write(OUT/'geometry_result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


class Action(record.Trial):
    def __init__(self,case):
        self.phase='approach';self.planned_s=0.;self.M8_streak=0;self.M8_lost=0
        self.handoff_time=None;self.action_end=None;self.events=[]
        super().__init__('M8_TRANSFER',case)
        with np.load(OUT/'planned_path.npz') as f:self.targets=f['ctrl_rad'].copy()
        if self.targets.shape!=(361,20) or not np.array_equal(self.targets[0],self.held):
            raise ValueError('M8 frozen entry/path mismatch')
        write(case/'initial_interface.json',dict(integration_max_difference=float(abs(self.raw-self.initial_state).max()),
            first_target_exact=True,feedback='simulation M8 contact/region gate only',no_physical_forward=True,
            root_applied=False,controller='q_hold + frozen shape delta'))

    def snapshot(self):
        super().snapshot();r=self.rows[-1];r['stage']=self.phase;r['planned_s']=self.planned_s
        r['M8_metrics']=module_metrics(self.m,self.diag,8,self.head)
        r['M9_metrics']=module_metrics(self.m,self.diag,9,self.head)
        world=np.zeros((2,21,3))
        for i,c in enumerate(self.diag.contact):
            f=np.zeros(6);mujoco.mj_contactForce(self.m,self.diag,i,f)
            fworld=c.frame.reshape(3,3).T@f[:3]
            b1,b2=[int(self.m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            for geom,body,sign in [(c.geom1,b2,1.),(c.geom2,b1,-1.)]:
                group=0 if geom in self.source else 1 if geom in self.target and c.pos[1]>.075 else None
                if group is not None and body in self.robot:world[group,body-self.head]+=sign*fworld
        r['contact_world_force_N']=world
        mm=r['M8_metrics'];valid=bool(mm[1]>=75. and mm[3]<=92. and r['contact_force_N'][1,8]>.1)
        r['M8_target_supported_region']=valid
        self.M8_streak=self.M8_streak+1 if valid else 0
        self.M8_lost=self.M8_lost+1 if self.handoff_time is not None and not valid else 0
        if valid and (len(self.rows)==1 or not self.rows[-2]['M8_target_supported_region']):
            self.events.append(dict(t_s=r['t_s'],event='M8_target_supported_region_entered',force_N=r['contact_force_N'][1,8]))

    def run_action(self):
        index=0;last=self.held.copy();outcome='action_window_exhausted'
        failure=self.danger();self.source_lost=self.target_lost=self.fast=0
        if failure:return ','.join(failure)
        for tick in range(round((ACTION_MAX+HOLD)/DT)):
            t=self.steps*.001
            if index<round(APPROACH/DT):index=tick
            elif index==round(APPROACH/DT) and tick>index:
                if self.M8_streak>=20:
                    self.handoff_time=t;self.phase='handoff';index+=1
                    self.events.append(dict(t_s=t,event='actual_M8_gate_allows_M7_M14_relief'))
                elif t>=APPROACH+WAIT_MAX:return 'M8_no_support_before_handoff_deadline'
                else:self.phase='wait_M8'
            elif index>round(APPROACH/DT) and index<360:index+=1
            if index==360 and self.action_end is None:
                self.action_end=t;self.phase='post_hold'
                self.events.append(dict(t_s=t,event='local_action_finished_post_hold_started'))
            self.planned_s=index*DT;target=self.targets[index]
            if not np.isfinite(target).all() or np.degrees(abs(target)).max()>88.+1e-10:return 'invalid_command'
            if np.degrees(abs(target-last)).max()>.5+1e-10:return 'command_rate_above10deg_s'
            self.d.ctrl[:]=target;last=target.copy()
            self.commands.append(dict(t_s=t,planned_s=self.planned_s,stage=self.phase,target_rad=target.copy(),applied=True))
            for _ in range(50):
                mujoco.mj_step(self.m,self.d);self.steps+=1
                if self.steps%10==0:
                    self.snapshot();failure=self.danger()
                    if self.M8_lost>=20:failure.append('M8_new_support_lost_for0p2s')
                    if failure:break
            if failure:return ','.join(failure)
            if self.action_end is not None and self.steps*.001>=self.action_end+HOLD-1e-9:
                return 'completed_action_and10s_hold'
            if tick%40==39:
                r=self.rows[-1]
                print(json.dumps(dict(t_s=r['t_s'],stage=self.phase,planned_s=self.planned_s,
                    M8_region=r['M8_target_supported_region'],M8_target_N=float(r['contact_force_N'][1,8]),
                    M8_xyz_radius_mm=r['M8_metrics'][:4].tolist(),contacts=np.count_nonzero(r['contact_force_N']>.1,axis=1).tolist())),flush=True)
        return outcome


def score(action,outcome):
    completed=outcome=='completed_action_and10s_hold';end=action.rows[-1]['t_s']
    shifted=[dict(r,t_s=r['t_s']-(end-12.)) for r in action.rows]
    grasp=record.holding(shifted,completed)
    grasp['completed_action_and10s_hold']=grasp.pop('completed_12s')
    result=dict(outcome=outcome,completed=completed,two_pipe_holding=grasp,handoff_time_s=action.handoff_time,
        action_end_s=action.action_end,events=action.events,physical_seconds=end,
        initial_M8_metrics=action.rows[0]['M8_metrics'],final_M8_metrics=action.rows[-1]['M8_metrics'],
        actual_new_target_modules=sorted({int(k) for r in action.rows for k in np.flatnonzero(r['contact_force_N'][1]>.1)
                                         if action.rows[0]['contact_force_N'][1,k]<=.1}),
        sampler_physical_state_max_difference=action.maximum_sampler_state_change,
        truth_assisted=True,hardware_feedback_claim=False,passed=False)
    if action.handoff_time is not None:
        after=[r for r in action.rows if r['t_s']>=action.handoff_time]
        before=[r for r in action.rows if action.handoff_time-.5<=r['t_s']<action.handoff_time]
        tail=[r for r in action.rows if action.action_end is not None and r['t_s']>=action.action_end]
        continuous=all(r['M8_target_supported_region'] for r in after)
        pre=np.mean([r['contact_world_force_N'] for r in before],axis=0)
        post=np.mean([r['contact_world_force_N'] for r in tail],axis=0) if tail else pre
        pre_normal=np.mean([r['contact_force_N'] for r in before],axis=0)
        post_normal=np.mean([r['contact_force_N'] for r in tail],axis=0) if tail else pre_normal
        result.update(M8_continuous_after_handoff=continuous,M8_mean_vertical_target_force_N=float(post[1,8,2]),
            M7_vertical_target_force_before_after_N=[float(pre[1,7,2]),float(post[1,7,2])],
            M14_vertical_source_force_before_after_N=[float(pre[0,14,2]),float(post[0,14,2])],
            M14_normal_source_force_before_after_N=[float(pre_normal[0,14]),float(post_normal[0,14])])
        redistribution=bool(post[1,8,2]>.1 and (pre[1,7,2]-post[1,7,2]>.1 or pre_normal[0,14]-post_normal[0,14]>.1))
        result['observed_local_vertical_support_redistribution']=redistribution
        result['passed']=bool(completed and grasp['passed'] and continuous and redistribution)
    return result


def run():
    admission=json.loads((OUT/'geometry_result.json').read_text(encoding='utf8'))
    if not admission['ready']:raise RuntimeError('Sole path not admitted; no dynamics')
    marker=OUT/'trial_ledger.csv';case=OUT/'actual'
    if marker.exists() or case.exists():raise FileExistsError('Sole trial already attempted; no retry')
    case.mkdir();shutil.copy2(__file__,OUT/'runner_source.py')
    def ledger(status,steps,outcome):
        with marker.open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['trial','review_attempt_number','status','physical_steps','physical_seconds','outcome'])
            w.writerow(['M8_TRANSFER',3,status,steps,steps*.001,outcome])
    ledger('ENTRY_STARTED',0,'pending');write(case/'entry_started.json',dict(counts_even_if_entry_fails=True,review_total_after_attempt=3))
    action=None;outcome='entry_exception';error=None
    try:
        action=Action(case);outcome=action.run_action()
    except BaseException as exc:error=repr(exc);outcome='entry_or_runtime_exception'
    finally:
        if action is not None:action.save()
    result=score(action,outcome) if action else dict(outcome=outcome,passed=False,physical_seconds=0)
    result.update(error=error,review_trials_used=3,remaining_trial_enabled=False,next='STOP, deliver actual action/obstruction for review')
    write(case/'result.json',result);ledger('FINISHED',action.steps if action else 0,outcome)
    if action is not None:
        with (OUT/'actual_M8_M9.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['t_s','stage','planned_s']+[f'M{k}_{x}' for k in [8,9] for x in
                ['x_mm','y_mm','z_mm','target_axis_mm','source_axis_mm','source_gap_mm','target_gap_mm','axis_x','axis_y','axis_z']]+
                ['M8_target_force_N','M7_target_vertical_N','M8_target_vertical_N','M14_source_vertical_N','M8_supported_region'])
            for r in action.rows:w.writerow([r['t_s'],r['stage'],r['planned_s'],*r['M8_metrics'],*r['M9_metrics'],
                r['contact_force_N'][1,8],r['contact_world_force_N'][1,7,2],r['contact_world_force_N'][1,8,2],
                r['contact_world_force_N'][0,14,2],r['M8_target_supported_region']])
        try:
            replay=[dict(r,sim_time=r['t_s'],psi_rad=np.radians(r['axial_roll_deg'][:8].mean())) for r in action.rows]
            record.replay_video(case,replay)
            cap=cv2.VideoCapture(str(case/'replay_1x.mp4'))
            write(case/'replay_metadata.json',dict(opened=cap.isOpened(),fps=cap.get(cv2.CAP_PROP_FPS),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),physical_seconds=result['physical_seconds'],speed=1.,
                source='actual saved qpos, not predicted root or reference animation'));cap.release()
        except Exception as exc:write(case/'replay_error.json',dict(error=repr(exc)))
    print(json.dumps(result,ensure_ascii=False,default=lambda x:x.tolist()),flush=True)


def replay_reference():
    """Saved private FK prefix only; never label it a physical trial."""
    folder=OUT/'reference';folder.mkdir(exist_ok=True)
    with np.load(OUT/'planned_path.npz') as f:
        times=f['time_s'].copy();poses=f['predicted_qpos'].copy()
    if len(times)!=len(poses) or not len(poses):raise ValueError('No saved FK prefix')
    if (folder/'reference_prefix_1x.mp4').exists():raise FileExistsError('Preserve existing reference replay')
    m=mujoco.MjModel.from_xml_path(str(record.MODEL));d=mujoco.MjData(m)
    renderer=mujoco.Renderer(m,height=480,width=960)
    camera=mujoco.MjvCamera();camera.lookat[:]=[1.03,.1,1.]
    camera.distance=1.3;camera.azimuth=150;camera.elevation=-20
    video=cv2.VideoWriter(str(folder/'reference_prefix_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(960,480))
    for t,q in zip(times,poses):
        d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
        frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        n=record.points(m,d);c=(n[8]+n[9])/2;r=np.linalg.norm(c[[0,2]]-[1100.,1000.])
        for y,text in [(24,'OFFLINE PARTIAL REFERENCE - NO PHYSICS RUN'),
                       (47,'Private FK root variable; not an applied robot/root motion'),
                       (70,'t=%.2fs  M8 y=%.2fmm radial=%.2fmm  (requires <=92mm)'%(t,c[1],r)),
                       (94,'Solver stopped at 6.05s; approach/handoff NOT completed')]:
            cv2.putText(frame,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,235,255),1)
        video.write(frame)
    video.release();renderer.close()
    cap=cv2.VideoCapture(str(folder/'reference_prefix_1x.mp4'))
    write(folder/'replay_metadata.json',dict(kind='OFFLINE_PARTIAL_REFERENCE_NOT_EXPERIMENT',
        physics_steps=0,source='saved private predicted_qpos',nominal_seconds=float(times[-1]),
        playback_speed=1.,fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        opened=cap.isOpened(),includes_private_root_variable=True))
    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,len(times)-1));ok,frame=cap.read()
    if ok:cv2.imwrite(str(folder/'reference_final.png'),frame)
    cap.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['plan','run','replay-reference'])
    args=parser.parse_args()
    if args.operation=='plan':plan()
    elif args.operation=='run':run()
    else:replay_reference()
