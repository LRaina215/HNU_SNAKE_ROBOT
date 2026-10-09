"""Active target coil hauls a passive suspended body through one shared shift."""
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from snake_control.suspended_supply_transfer import WholeCurveMapper,desired_curve,nodes,LINK,DT,LIMIT,STEP

class PassiveSupplyRoll:
    def __init__(self,model,data,entry_gain=.18,speed=.0025,preserve_front=False):
        self.m=model;self.mapper=WholeCurveMapper(model);self.qa=self.mapper.qa;self.va=self.mapper.va
        self.last=data.ctrl.copy();self.shift=0.;self.radius_delta=0.;self.speed=speed;self.entry_gain=entry_gain
        actual=nodes(data);nominal,v=desired_curve('peel',0.)
        self.initial_field=CubicSpline(v[::-1],(actual-nominal)[::-1],axis=0,bc_type='clamped')
        self.last_guide=actual.copy();self.origin=actual.copy();self.kp=np.full(20,127.);self.preserve_front=preserve_front;self.engaged=set()
        self.damping=model.dof_damping[self.va].copy();self.received_age=np.zeros(21);self.received=set()
        self.phi0=np.arctan2((.5*(actual[:-1]+actual[1:]))[:,2]-1.,(.5*(actual[:-1]+actual[1:]))[:,0]-1.1)
        self.phi=self.phi0.copy();self.previous_raw=self.phi0.copy();self.last_progress=0.;self.best_angular=0.;self.best_received=0;self.failure=None

    def observe(self,points,forces,dt):
        c=.5*(points[:-1]+points[1:]);rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        raw=np.arctan2(c[:,2]-1.,c[:,0]-1.1)
        self.phi+=np.arctan2(np.sin(raw-self.previous_raw),np.cos(raw-self.previous_raw));self.previous_raw=raw
        winding=self.phi0-self.phi
        # Distinguish retained new wrapping from isolated contact and axial slide.
        qualifying=(rr<.112)&(rr>.065)&(c[:,1]>.15)&(winding>.35)&(forces[1].sum()>3.)
        self.received_age=np.where(qualifying,self.received_age+dt,0.)
        self.received={j for j in range(9,21) if self.received_age[j]>=1.}
        return c,rr,winding

    def command(self,data,t,forces):
        p=nodes(data);c,rr,winding=self.observe(p,forces,DT)
        if self.preserve_front:self.engaged.update(self.received)
        prefix=9
        while prefix in (self.engaged if self.preserve_front else self.received):prefix+=1
        # Restore only at the physical intake. Nominal shift alone cannot turn
        # an airborne posterior joint into a full-power winding joint.
        desired_kp=np.zeros(20);desired_kp[:min(20,prefix-1)]=127.
        entry=[]
        for index in range(max(8,prefix-1),min(20,prefix+1)):
            module=index+1
            proximity=np.clip((.205-rr[module])/.08,0.,1.)
            desired_kp[index]=127.*self.entry_gain*proximity
            entry.append(index)
        ramp=np.clip(t/1.5,0.,1.)
        desired_kp=(1-ramp)*127.+ramp*desired_kp
        self.kp+=np.clip(desired_kp-self.kp,-127.*DT/1.5,127.*DT/1.5)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        active=np.zeros(20,dtype=bool);active[:min(20,prefix-1)]=True
        active[entry]=np.array([self.kp[j]>.5 for j in entry])
        error=float(np.degrees(np.abs(self.last[active]-data.qpos[self.qa][active])).max())
        stress=float(np.mean(np.abs(data.qfrc_actuator[self.va][active])>2.2))
        scale=float(np.clip((15.-error)/12.,.15,1.)*np.clip(1.-stress,.4,1.))
        angular=float(max(0.,winding[9: min(12,21)].max()))
        if len(self.received)>self.best_received or angular>self.best_angular+.08:
            self.best_angular=max(self.best_angular,angular);self.best_received=max(self.best_received,len(self.received));self.last_progress=t
        normal=float(forces[1].sum())
        self.radius_delta=float(np.clip(self.radius_delta+np.clip(.000012*(normal-50.),-.00025,.00025)*DT,-.0025,.002))
        proposal=min(12*LINK,self.shift+self.speed*DT*scale)
        guide,v=desired_curve('haul',1.,proposal,self.radius_delta,self.initial_field)
        if self.preserve_front:
            # Carry the measured front coil through a common screw displacement.
            # Advecting a nonuniform residual field distorted its wrap in P01.
            fraction=.180/np.hypot(2*np.pi*.0915,.180);beta=np.sqrt(1-fraction*fraction)
            angle=beta*proposal/.0915;cs,sn=np.cos(angle),np.sin(angle)
            rot=np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])
            pivot=np.array([1.1,0.,1.]);captured=(self.origin-pivot)@rot.T+pivot
            captured[:,1]+=fraction*proposal
            mouth=self.origin[9];phi=np.arctan2(mouth[2]-1.,mouth[0]-1.1)
            i=np.arange(22);phase=phi+beta*(i-9)*LINK/.0915-angle
            ideal=np.column_stack([1.1+(.0915+self.radius_delta)*np.cos(phase),
                mouth[1]+fraction*(proposal-(i-9)*LINK),1.+(.0915+self.radius_delta)*np.sin(phase)])
            fade=np.clip(proposal/(1.5*LINK),0,1);fade=fade*fade*(3-2*fade)
            guide=captured.copy();guide[10:]=(1-fade)*captured[10:]+fade*ideal[10:]
        # Released supply follows its actual curve. Only the active prefix and
        # receiving mouth enter the position objective; passive q is observed.
        weights=np.zeros(22);last_active=int(np.flatnonzero(active)[-1])+1
        weights[:prefix+1]=2.;weights[prefix+1:last_active+2]=.8
        guide[last_active+2:]=p[last_active+2:]
        _,candidate,fit=self.mapper.solve(guide,self.last_guide,data,forces,self.last,max_nfev=20,active_mask=active,node_weights=weights)
        target=self.last+np.clip(np.clip(data.qpos[self.qa],-LIMIT,LIMIT)-self.last,-STEP,STEP)
        target[active]=candidate[active]
        self.shift=proposal;self.last=target.copy();self.last_guide=guide.copy()
        if self.shift>=2*LINK and not self.received and t-self.last_progress>20.:self.failure='full_two_rod_action_no_retained_new_wrap'
        if self.shift>=3*LINK and not self.received:self.failure='three_rod_reference_opportunity_without_new_wrap'
        if t-self.last_progress>60. and self.shift>=2*LINK:self.failure='no_actual_new_winding_60s'
        regions=np.where(self.kp<.5,2,np.where(self.kp>126.,0,1))
        info=dict(stage='active_coil_passive_supply',shape_parameter=0.,shift_mm=self.shift*1000,
            shift_speed_mm_s=self.speed*scale*1000,tracking_deg=error,curve_radius_delta_mm=self.radius_delta*1000,
            reference_points_mm=guide*1000,kp_nm_per_rad=self.kp.copy(),joint_damping_nm_s_per_rad=self.damping,
            active_joint_mask=active,region_codes=regions,actual_new_retained_modules=sorted(self.received),
            actual_new_winding_rad=winding,actual_radial_mm=rr*1000,normal_force_n=normal,failure=self.failure,
            preserve_measured_front_screw_shape=self.preserve_front,engaged_modules=sorted(self.engaged),
            actual_rear_position_objective=False,small_prediction_residual_freezes_group=False,mapper=fit,
            maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()))
        return target,info
