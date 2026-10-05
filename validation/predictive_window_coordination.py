"""Experimental short-horizon selection between existing compliant windows.

Only the causal private dynamics belief is rolled out. The physical simulator
is absent from this API. Every candidate retains A-D/Yamada and the same
position delivery; selection changes rolling rate and the release budget.
The numerical score is an engineering hypothesis, not a stability proof.
"""
import copy
from dataclasses import replace
import time
import numpy as np
import mujoco


def clone_state(obj,shared):
    result=copy.copy(obj)
    for key,value in vars(obj).items():
        setattr(result,key,value if key in shared else copy.deepcopy(value))
    return result


class PredictiveWindowCoordination:
    def __init__(self,horizon=1.2,period=.6):
        self.horizon=horizon;self.period=period;self.remaining=0.
        self.rate=1.;self.conserve=True;self.last_report={}

    @staticmethod
    def observe(model,data,tracker,geometry,controller):
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,data.qpos[3:7]);rotation=rotation.reshape(3,3)
        world=np.vstack([data.site('s_head').xpos,
            *[data.joint('joint'+str(i)).xanchor for i in range(1,21)],data.site('s_tail').xpos])
        points=1000*(world-data.qpos[:3])@rotation
        controller.corner=rotation.T@(tracker.fixed['corner_mm']-1000*data.qpos[:3])
        controller.source=rotation.T@tracker.fixed['source'];controller.target=rotation.T@tracker.fixed['target']
        controller.body_axis=-controller.target
        support=np.zeros((3,21))
        for j in range(data.ncon):
            contact=data.contact[j]
            for axis,slot in enumerate(tracker.pipe_slots):
                if slot not in (contact.geom1,contact.geom2):continue
                other=contact.geom2 if contact.geom1==slot else contact.geom1
                module=tracker.body_to_module.get(int(model.geom_bodyid[other]))
                if module is None:continue
                wrench=np.zeros(6);mujoco.mj_contactForce(model,data,j,wrench)
                support[axis,module]+=max(0.,wrench[0])
        return dict(geometry,points_mm=points),support,(world[:-1]+world[1:])/2

    @staticmethod
    def acquisition_value(controller,points):
        centers=(points[:-1]+points[1:])/2;rel=centers-controller.corner
        radial=np.linalg.norm(np.cross(rel,controller.target),axis=1)
        gap=np.maximum(0.,radial-controller.radius-controller.options.shell_radius_mm)
        axial=rel@controller.target
        # Smooth measured association of all modules; changing a software
        # boundary cannot earn progress. A free end far from the target earns
        # essentially no credit, even if its target reference is a helix.
        nearby=np.exp(-(gap/controller.link)**2)
        forward=1/(1+np.exp(-np.clip(axial/(.5*controller.link),-40,40)))
        return float(np.sum(nearby*forward))

    def rollout(self,tracker,controller,delivery,geometry,measured,history,rate,conserve,dt,rear_rate=1.):
        m=tracker.observer.model;d=copy.copy(tracker.observer.data)
        d.qpos[tracker.qadr]=measured;mujoco.mj_forward(m,d)
        c=clone_state(controller,{'forward_points'});v=clone_state(delivery,{'point_jacobians'})
        c.window=replace(c.window,conserve_middle=conserve)
        c.rear_motion_scale=rear_rate
        h=[np.array(x).copy() for x in history]
        g,support,centers=self.observe(m,d,tracker,geometry,c)
        initial_value=self.acquisition_value(c,g['points_mm'])
        initial_wrap=c.support(g['points_mm'],h)['front_wrap'];minimum_wrap=initial_wrap
        maximum_speed=0.;maximum_joint=0.;no_support=0;sample_count=0
        for step in range(round(self.horizon/dt)):
            if step:
                g,support,centers=self.observe(m,d,tracker,geometry,c)
                h.extend([support.copy(),support.copy()]);del h[:-60]
            q=d.qpos[tracker.qadr].copy();origin=d.ctrl.copy()
            target,info=c.compute(g,q,c.requested_phase-.1875*dt*rate,dt,h)
            for sub in range(1,round(dt/v.period)+1):
                if sub>1:
                    g,support,centers=self.observe(m,d,tracker,geometry,c)
                    h.extend([support.copy(),support.copy()]);del h[:-60]
                measured_q=d.qpos[tracker.qadr].copy()
                cmd,_=v.compute(origin+(target-origin)*sub/round(dt/v.period),
                    (target-origin)/dt,measured_q,d.ctrl.copy(),np.ones(c.n))
                d.ctrl[:]=cmd
                # Score at the same 10-ms cadence as the physical audit.
                for _ in range(round(v.period/.01)):
                    before=centers.copy();mujoco.mj_step(m,d,nstep=round(.01/m.opt.timestep));mujoco.mj_forward(m,d)
                    g,support,centers=self.observe(m,d,tracker,geometry,c)
                    maximum_speed=max(maximum_speed,float(np.linalg.norm((centers-before)/.01,axis=1).max()))
                    maximum_joint=max(maximum_joint,float(np.degrees(abs(d.qvel[tracker.observer.vadr])).max()))
                    no_support+=int(not np.any(support>.1));sample_count+=1
            c.last_target=d.ctrl.copy()
            minimum_wrap=min(minimum_wrap,c.support(g['points_mm'],h)['front_wrap'])
        final_value=self.acquisition_value(c,g['points_mm']);final_wrap=c.support(g['points_mm'],h)['front_wrap']
        risk=max(0.,maximum_speed/.4-1)+max(0.,maximum_joint/90.-1)+no_support/max(sample_count,1)
        # Prefer acquisition without unwinding existing front support. Safety
        # violations rank first; the 0.4 m/s prediction margin precedes the
        # independent, unchanged 0.5 m/s physical acceptance threshold.
        score=-(final_value-initial_value)+3*max(0.,initial_wrap-final_wrap)+.01*maximum_speed/.2
        return dict(rate=rate,rear_rate=rear_rate,conserve=conserve,risk=risk,score=score,
            module_peak_m_s=maximum_speed,joint_peak_deg_s=maximum_joint,
            acquisition_change=final_value-initial_value,wrap_before=initial_wrap,
            wrap_after=final_wrap,minimum_wrap=minimum_wrap,no_support_samples=no_support)

    def choose(self,tracker,controller,delivery,geometry,measured,history,dt):
        fresh=self.remaining<=1e-9
        if fresh:
            started=time.perf_counter();trials=[]
            for rate in (1.,.25,0.):
                for conserve in (False,True):
                    trials.append(self.rollout(tracker,controller,delivery,geometry,measured,history,rate,conserve,dt))
            selected=min(trials,key=lambda x:(x['risk'],x['score'],-x['rate']))
            self.rate=selected['rate'];self.conserve=selected['conserve'];self.remaining=self.period
            self.last_report=dict(candidates=trials,selected=selected,computation_seconds=time.perf_counter()-started,
                inputs='causal private dynamics belief; no physical pose, axes or contact labels')
        self.remaining-=dt
        controller.window=replace(controller.window,conserve_middle=self.conserve)
        return controller.requested_phase-.1875*dt*self.rate,dict(predictive_coordination=dict(
            fresh=fresh,rate=self.rate,conserve=self.conserve,**(self.last_report if fresh else {})))


class PredictiveSupportGuard(PredictiveWindowCoordination):
    """Leave nominal motion intact unless inferred dynamics predicts a swing.

    The earlier common-rate search could not alter the relative pipe-phase
    motion at source detachment. This diagnostic also admits held/reversed rear
    rolling. It never disables the material budget or changes A-D, formation,
    actuator limits or physical support. Prediction is not a safety guarantee.
    """
    def __init__(self):
        super().__init__(horizon=2.,period=.4)
        self.rear_rate=1.
        self.phase_delay=0.

    def choose(self,tracker,controller,delivery,geometry,measured,history,dt,nominal_phase):
        fresh=self.remaining<=1e-9
        if fresh:
            started=time.perf_counter()
            nominal=self.rollout(tracker,controller,delivery,geometry,measured,history,1.,True,dt,1.)
            trials=[nominal]
            if nominal['risk']>0:
                for rate,rear in ((1.,0.),(1.,-1.),(.25,1.),(0.,0.)):
                    trials.append(self.rollout(tracker,controller,delivery,geometry,measured,history,rate,True,dt,rear))
            selected=min(trials,key=lambda x:(x['risk'],x['score'],-x['rate']))
            self.rate=selected['rate'];self.rear_rate=selected['rear_rate'];self.remaining=self.period
            self.last_report=dict(candidates=trials,selected=selected,computation_seconds=time.perf_counter()-started,
                inputs='causal private dynamics belief; no physical pose, axes or contact labels',
                guard_active=nominal['risk']>0)
        self.remaining-=dt
        controller.rear_motion_scale=self.rear_rate
        # Retain the caller's original k*dt arithmetic when no intervention
        # occurred. Repeated subtraction changed phase by ~4e-15 on step 2;
        # contact/optimisation sensitivity amplified that into a new trajectory
        # before the guard ever acted. Only interventions accumulate delay.
        self.phase_delay+=(1-self.rate)*.1875*dt
        return nominal_phase+self.phase_delay,dict(predictive_coordination=dict(
            fresh=fresh,rate=self.rate,rear_rate=self.rear_rate,conserve=True,
            **(self.last_report if fresh else {})))
