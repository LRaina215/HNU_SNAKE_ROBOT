"""Experimental material-phase windows with contact-guided body recruitment.

The input axes and contact identities are simulation truth in stage 1.
Joint targets retain measured Bishop/Yamada bending and the original A-D
compliance difference. No physical servo or contact parameters are modified.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import math
import numpy as np
from traveling_compliant_winding import TravelingCompliantWinding
from three_segment_incremental import angles_from_integrals
from three_segment_compliance import LocalSection
from segmented_pole_backbone import smoothstep


def rotate_rows(field, angles):
    c,s=np.cos(angles),np.sin(angles)
    return np.column_stack((c*field[:,0]-s*field[:,1],s*field[:,0]+c*field[:,1]))


@dataclass(frozen=True)
class WindowOptions:
    joints_per_rad: float = .4/.1875
    maximum_lead: float = 2.
    target_lead_deg: float = 8.
    tail_mode: str = 'support'
    finish_mode: str = 'original'
    supported_recovery: float = .1
    rear_phase_lead: float = 0.
    support_centers: bool = False
    rolling_lookahead_seconds: Optional[float] = None
    section_domain: bool = False


class WindowCoordinatedWinding(TravelingCompliantWinding):
    def __init__(self,*args,window=WindowOptions(),**kwargs):
        super().__init__(*args,**kwargs)
        self.window=window
        self.requested_phase=self.phase
        self.rear_phase=self.phase
        self.joint_phase=np.zeros(self.n)
        self.previous_measured=np.array(kwargs['measured'])
        self.command_before_last=self.last_target.copy()
        self.follow=1.
        self.feed=0.
        self.rear_motion_scale=1.

    def rear_release_permitted(self):
        return True

    def transfer_region(self,positions):
        ds,dr,ps,pt=self.distances(positions)
        return (dr+10<ds)&(dr<self.radius+29+15)&(pt>self.radius/2)

    def handoff_geometry_ready(self,points,history,status):
        source_clear=len(history)>=60 and not np.any(np.asarray(history)[-60:,0]>.1)
        cleared=np.min((points-self.corner)@self.target)>self.radius+self.options.shell_radius_mm
        return bool(source_clear and cleared and status['target_contact']),bool(source_clear),bool(cleared)

    def release_geometry_ready(self,points,prefix):
        _,_,ps,_=self.distances(points[1:-1])
        return self.rear<self.n and ps[self.rear]>-self.formation.release_distance_links*self.link

    def rolling_observation(self,geometry,measured,observed):
        rolling,info=super().rolling_observation(geometry,measured,observed)
        # The initially draped front also needs shape recovery while rolling.
        # Compare its old 0.1 gain with the 0.35 used by recruited joints;
        # this adjusts the position reference, not actuator strength.
        gain=info['formation_gain'].copy()
        gain[:self.initial_front]=self.window.supported_recovery*float(smoothstep(self.elapsed/1.6))
        change=gain[:,None]*(info['formation_nominal']-observed)
        info.update(formation_gain=gain,formation_change=change)
        return observed+change,info

    def support(self,points,history):
        centers=(points[:-1]+points[1:])/2
        # Shell contact and the independent transfer audit use module centres.
        # A polygonal helix's joint vertices can lie outside the same radius
        # band even while its links are already supported by the target pipe.
        positions=centers[:self.n] if self.window.support_centers else points[1:-1]
        region=self.transfer_region(positions)
        prefix=self.initial_front
        while prefix<self.n and region[prefix]:prefix+=1
        h=np.eye(3)[np.argmin(abs(self.target))]
        h-=h@self.target*self.target;h/=np.linalg.norm(h)
        v=np.cross(self.target,h)
        radial=centers-self.corner
        angles=np.arctan2(radial@v,radial@h)
        forces=np.asarray(history)
        contacts=np.mean(forces[-40:,1]>.1,axis=0)>.25
        ids=np.flatnonzero(contacts & (np.arange(self.n+1)<=prefix))
        theta=np.sort(np.mod(angles[ids],2*np.pi))
        span=2*np.pi-np.max(np.diff(np.r_[theta,theta[0]+2*np.pi])) if len(theta)>1 else 0.
        a=np.unwrap(np.arctan2((points[:prefix+2]-self.corner)@v,(points[:prefix+2]-self.corner)@h))
        wrap=abs(a[-1]-a[0])/(2*np.pi)
        contact=bool(len(ids)>0)
        # Angular contact coverage and measured wrap, not just any contact.
        grip=float(np.clip((span-np.pi/2)/(np.pi/2),0,1)*np.clip((wrap-.5)/.5,0,1))
        return dict(actual_transfer=region,actual_prefix=prefix,target_contact=contact,
            contact_span_deg=float(np.degrees(span)),front_wrap=float(wrap),grip=grip,
            source_contact=bool(np.any(forces[-1,0]>.1)))

    def recruit(self,points,history,status,front_delta,dt):
        prefix=status['actual_prefix'];old=[self.front,self.rear]
        ready,source_clear,cleared=self.handoff_geometry_ready(points,history,status)
        if ready:self.handed_off=True
        limit=min(self.n,self.rear,self.n if self.handed_off else prefix+self.window.maximum_lead)
        if status['target_contact']:
            self.progress=max(self.progress,min(limit,self.progress+abs(front_delta)*self.window.joints_per_rad*self.follow))
        self.formation_weights=smoothstep(np.clip(self.progress-np.arange(self.n),0,1))
        self.front=min(self.n,int(math.ceil(self.progress-1e-9)))
        room=self.rear-self.front<self.middle_max
        remaining=np.any(np.asarray(history)[-1,0,self.rear+2:]>.1)
        near=self.release_geometry_ready(points,prefix)
        reserve=max(0,6-max(0,prefix-8))
        if self.handed_off:near=self.rear<self.n;reserve=0
        release=(self.progress>self.initial_front+.5 and status['target_contact'] and room and near
                 and (remaining or prefix>=12 or self.handed_off) and self.n-self.rear>reserve
                 and self.rear_release_permitted())
        # The same supply factor gates phase advance and boundary release.
        self.rear_ready=self.rear_ready+dt*self.feed if release else 0.
        if self.rear_ready>=.6-1e-9:
            self.middle_reference[self.rear]=self.last_target[self.rear]
            self.rear+=1;self.rear_ready=0.
        if old!=[self.front,self.rear]:
            self.events.append(dict(time=self.elapsed,old=old,new=[self.front,self.rear],actual_prefix=prefix,
                                    formation_progress=self.progress,release=bool(release),feed=self.feed))
        return dict(formation_progress=self.progress,release_allowed=bool(release),rear_reserve=reserve,
            handed_off=self.handed_off,original_pipe_clear=bool(source_clear),junction_cleared=bool(cleared))

    def update_middle_reference(self,points,measured,dt):
        middle=self.weights[:,1]>0
        self.middle_reference[middle]+=-math.expm1(-dt/self.options.follow_seconds)*(measured[middle]-self.middle_reference[middle])
        return {}

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        previous=self.last_target.copy();self.elapsed+=dt;self.current_dt=dt
        points=np.asarray(geometry['points_mm'])
        status=self.support(points,contact_window)
        change=previous-self.command_before_last
        motion=measured-self.previous_measured
        front_slice=slice(0,max(self.initial_front,self.front))
        norm=float(change[front_slice]@change[front_slice])
        following=float(np.clip(motion[front_slice]@change[front_slice]/norm,0,1)) if norm>1e-8 else 1.
        self.follow+=(1-math.exp(-dt/.4))*(following-self.follow)
        demand=phase-self.requested_phase;self.requested_phase=phase
        acquired=float(np.clip(status['actual_prefix']-self.initial_front,0,1))
        supply=status['grip']*acquired
        if self.handed_off:supply=1.
        if self.window.tail_mode=='same':supply=1.
        if self.window.tail_mode=='hold':supply=0. if not self.handed_off else 1.
        if self.window.tail_mode=='coupled':supply=.25+.75*supply
        self.feed+=(1-math.exp(-dt/.6))*(supply-self.feed)
        front_delta=demand
        rear_delta=demand*self.feed*self.rear_motion_scale
        if self.window.tail_mode=='coupled':
            # Preserve the captured two-axis phase relation within a bounded
            # lead. A stopped rear must not let the front accumulate cycles
            # of relative rotation in the connecting body.
            lag=self.rear_phase-self.phase
            lead_room=float(np.clip(1-lag/(np.pi/4),0,1))
            front_delta=demand*min(1.,self.feed+lead_room)
        if self.window.rear_phase_lead and acquired:
            # A bounded extra rear rotation tests the OTHER exit orientation,
            # rather than accumulating an unbounded front lead. This is an
            # experimental material-phase offset, not a claimed CMU mapping.
            error=self.window.rear_phase_lead-(self.phase-self.rear_phase)
            rear_delta-=float(np.clip(error,-abs(demand),abs(demand)))
        previous_phase=self.phase
        self.phase+=front_delta;self.rear_phase+=rear_delta
        recruitment=self.recruit(points,contact_window,status,front_delta,dt)
        desired=self.desired_weights();difference=desired-self.weights
        self.weights+=difference*np.minimum(1.,dt/self.options.switch_seconds/np.maximum(abs(difference).max(axis=1),1e-12))[:,None]
        middle_info=self.update_middle_reference(points,measured,dt)
        observed=self.material_integrals(points,measured)
        active=desired[:,1]==0;entering=active&~self.active
        if np.any(entering):
            nominal=self.material_integrals(self.forward_points(previous),previous)
            self.preload[entering]=rotate_rows(nominal-observed,-self.joint_phase)[entering]
        self.active=active
        rolling,shape=self.rolling_observation(geometry,measured,observed)
        delta=self.weights[:,0]*front_delta+self.weights[:,2]*rear_delta
        self.joint_phase+=delta
        # Changing feedback frequency must not silently shrink the position
        # lead that produces rolling torque. Phase/state and recruitment still
        # integrate dt; the measured-field target may use a fixed lookahead.
        lookahead=1. if self.window.rolling_lookahead_seconds is None else self.window.rolling_lookahead_seconds/dt
        rolled=angles_from_integrals(rotate_rows(rolling,delta*lookahead)+rotate_rows(self.preload,self.joint_phase))
        compliance=np.zeros(self.n);section_domains=[]
        for column,axis,step in ((0,self.target,front_delta),(2,self.source,rear_delta)):
            ids=np.flatnonzero(self.weights[:,column]>1e-12)
            if not len(ids):continue
            lo,hi=int(ids[0]),int(ids[-1])+1
            if self.window.section_domain:
                from section_compliance_domain import correction
                value,domain=correction(geometry,self.zero_parameters,self.parameters,lo,hi,axis,self.corner,measured,step*lookahead)
                compliance[lo:hi]=value;section_domains.append(dict(column=column,**domain))
                continue
            zero=LocalSection(geometry,self.zero_parameters,lo,hi,axis,self.corner)
            shaped=LocalSection(geometry,self.parameters,lo,hi,axis,self.corner)
            gauge=zero.align(measured[lo:hi],0.)
            compliance[lo:hi]=shaped.angles(gauge+step*lookahead)-zero.angles(gauge+step*lookahead)
        raw=(self.weights[:,0]+self.weights[:,2])*(rolled+compliance)+self.weights[:,1]*self.middle_reference
        finish={}
        if self.handed_off and self.progress>=self.n and self.window.finish_mode=='original':
            local=LocalSection(geometry,self.parameters,0,self.n,self.target,self.corner)
            if self.finish_elapsed is None:
                self.finish_elapsed=0.;self.finish_offset=local.align(previous,previous_phase)
                self.finish_residual=previous-local.angles(previous_phase+self.finish_offset)
            self.finish_elapsed+=dt;fraction=float(smoothstep(self.finish_elapsed/2.))
            raw=local.angles(self.phase+self.finish_offset)+(1-fraction)*self.finish_residual
            rel=points[1:-1]-self.corner
            radius=float(np.median(np.linalg.norm(np.cross(rel[4:-4],self.target),axis=1)))
            a,b=self.helix_basis(radius,self.pitch_sign)
            fit=np.linalg.lstsq(np.column_stack([a,b])[4:-4],raw[4:-4],rcond=None)[0]
            angle=math.atan2(fit[1],fit[0]);nominal=a*np.cos(angle)+b*np.sin(angle)
            ends=np.r_[np.arange(4),np.arange(self.n-4,self.n)]
            raw[ends]+=fraction*self.formation.end_recovery*(nominal[ends]-measured[ends])
            finish=dict(finish_local=True,finish_elapsed=self.finish_elapsed)
        budget=math.radians(self.window.target_lead_deg)
        conditioned=measured+np.clip(raw-measured,-budget,budget)
        target=previous+np.clip(conditioned-previous,-self.options.maximum_speed_rad_s*dt,self.options.maximum_speed_rad_s*dt)
        signs=np.where(np.arange(self.n)%2,-1.,1.)
        low,high=np.sort(np.radians([[j['minimum_deg'],j['maximum_deg']] for j in geometry['config']['joints']])*signs[:,None],axis=1).T
        target=np.clip(target,low+.01,high-.01)
        self.command_before_last=previous;self.previous_measured=measured.copy();self.last_target=target.copy()
        return target,dict(front=self.front,rear=self.rear,weights=self.weights.copy(),raw=raw,
            measured=measured,front_phase=self.phase,rear_phase=self.rear_phase,phase_difference=self.rear_phase-self.phase,
            follow=self.follow,feed=self.feed,front_delta=front_delta,rear_delta=rear_delta,
            compliance_delta=compliance,section_domains=section_domains,**shape,**status,**recruitment,**middle_info,**finish)

    def configuration(self):
        return dict(**super().configuration(),window=asdict(self.window))
