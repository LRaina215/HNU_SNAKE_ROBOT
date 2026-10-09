"""A growing horizontal-pipe helix, driven by a common material roll phase."""
import numpy as np
from snake_control.suspended_transfer_route import receiving,ALPHA,BETA,R,INTAKE
from snake_control.suspended_supply_transfer import LINK,DT,LIMIT,STEP,nodes
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone

def backbone(shift):
    material=9*LINK+shift-np.arange(22)*LINK
    return receiving(material),material

class HorizontalPipeRoll:
    def __init__(self,model,data,phase_sign=1.,speed=.0025,front_passive=False):
        self.m=model;self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]];self.va=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.last=data.ctrl.copy();self.shift=0.;self.phase=0.;self.sign=phase_sign;self.speed=speed
        self.initial=data.qpos[self.qa].copy();self.kp=np.full(20,127.);self.received_age=np.zeros(21);self.received=set()
        p,_=backbone(0);bone=SpatialBishopBackbone(p*1000,np.full(21,LINK*1000))
        self.nominal0=bone.joint_angles();self.residual=self.initial-self.nominal0
        self.phi0=np.arctan2((.5*(nodes(data)[:-1]+nodes(data)[1:]))[:,2]-1.,(.5*(nodes(data)[:-1]+nodes(data)[1:]))[:,0]-1.1)
        self.phi=self.phi0.copy();self.previous_raw=self.phi0.copy();self.engaged=set();self.best_received=0;self.last_progress=0.;self.failure=None
        self.front_passive=front_passive

    def command(self,data,t,forces):
        p=nodes(data);c=.5*(p[:-1]+p[1:]);rr=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        raw=np.arctan2(c[:,2]-1,c[:,0]-1.1);self.phi+=np.arctan2(np.sin(raw-self.previous_raw),np.cos(raw-self.previous_raw));self.previous_raw=raw
        winding=self.phi0-self.phi
        near=(rr<.12)&(rr>.065)&(c[:,1]>.15)
        self.received_age=np.where(near,self.received_age+DT,0.)
        self.received={j for j in range(9,21) if self.received_age[j]>=1. and winding[j]>.35}
        if len(self.received)>self.best_received:self.best_received=len(self.received);self.last_progress=t
        self.engaged.update(self.received)
        error=float(np.degrees(abs(self.last-data.qpos[self.qa])).max())
        scale=float(np.clip((20.-error)/15.,.2,1.))
        increment=self.speed*DT*scale;self.shift=min(12*LINK,self.shift+increment)
        # One phase controls every active curvature component. Discrete hinge
        # targets are produced from the growing route, not absolute world fitting.
        self.phase+=self.sign*increment/(.029/BETA)
        guide,material=backbone(self.shift)
        bone=SpatialBishopBackbone(guide*1000,np.full(21,LINK*1000))
        q=bone.joint_angles(self.phase)
        # Preserve initial reconstruction discrepancy only in the already
        # wrapped prefix; taper it as material advances one complete turn.
        fade=max(0.,1-self.shift/(2*np.pi*R/BETA))
        q[:8]+=fade*self.residual[:8]*np.cos(self.phase)
        q=np.clip(q,-LIMIT,LIMIT)
        if self.front_passive:
            prefix=9
            while prefix in self.engaged:prefix+=1
            kp=np.zeros(20);kp[:min(20,prefix-1)]=127.
            kp[max(8,prefix-1):min(20,prefix+1)]=63.5
            ramp=np.clip(t/1.5,0,1);kp=(1-ramp)*127+ramp*kp
            self.kp+=np.clip(kp-self.kp,-127*DT/1.5,127*DT/1.5)
            q[self.kp<.5]=np.clip(data.qpos[self.qa][self.kp<.5],-LIMIT,LIMIT)
        self.m.actuator_gainprm[:,0]=self.kp;self.m.actuator_biasprm[:,1]=-self.kp
        target=self.last+np.clip(q-self.last,-STEP,STEP);self.last=target.copy()
        if self.shift>=2*LINK and not self.received and t>90:self.failure='two_rod_action_without_retained_intake'
        if t-self.last_progress>90 and self.shift>=3*LINK:self.failure='no_new_retained_intake_90s'
        info=dict(stage='horizontal_pipe_material_roll',shape_parameter=self.phase,shift_mm=self.shift*1000,tracking_deg=error,
            curve_radius_delta_mm=0.,reference_points_mm=guide*1000,kp_nm_per_rad=self.kp.copy(),active_joint_mask=self.kp>.5,
            region_codes=np.where(self.kp<.5,2,np.where(self.kp>126,0,1)),actual_new_winding_rad=winding,
            actual_new_retained_modules=sorted(self.received),material_coordinate_mm=material*1000,phase_rad=self.phase,
            phase_sign=self.sign,desired_joint_rad=q.copy(),nominal_bishop_joint_rad=bone.joint_angles(),
            maximum_dispatch_step_deg=float(np.degrees(abs(target-data.ctrl)).max()),failure=self.failure)
        return target,info
