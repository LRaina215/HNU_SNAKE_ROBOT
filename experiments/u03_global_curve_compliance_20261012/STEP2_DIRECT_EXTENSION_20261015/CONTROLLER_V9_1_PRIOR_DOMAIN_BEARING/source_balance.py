"""Local source C counterbalances the actual unsupported external moment.

The known source X-axis is fixed. Robot masses/poses and unlabeled loads are
allowed inputs. This is a posterior curvature deformation, not feed or rolling.
"""
import numpy as np
from physics import mujoco
from contact_sensor import known_source

class SourceBalance:
    def __init__(self,m,source,settings):
        self.m=m;self.source=source;self.p=settings;self.head=m.body('head').id
        self.va=m.jnt_dofadr[m.actuator_trnid[:,0]];self.offset=np.zeros(20);self.baseline=0.;self.baseline_ready=False
    def measure(self,d,sensor):
        origin=np.r_[0.,self.source['center_yz_m']]
        moment=sum(np.cross(d.xipos[self.head+j]-origin,self.m.body_mass[self.head+j]*self.m.opt.gravity)[0] for j in range(21))
        for ct in sensor['contacts']:
            if not known_source(ct,self.source):moment+=np.cross(ct['point']-origin,ct['world_force_N'])[0]
        return float(moment)
    def update(self,d,sensor,dt,active,first_capture,t):
        p=self.p;moment=self.measure(d,sensor);gradient=np.zeros(20);anchor=-1;rate=np.zeros(20)
        contacts=[ct for ct in sensor['contacts'] if known_source(ct,self.source) and ct['normal_N']>.5]
        if not active:self.baseline=moment;self.baseline_ready=True
        if active and dt>0 and contacts and first_capture is None:
            anchor=max(contacts,key=lambda ct:ct['normal_N'])['material']
            jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(self.m,d,jp,jr,self.head+anchor)
            J=np.vstack([jp,jr]);root=-np.linalg.solve(J[:,:6],J[:,self.va])
            for j in range(21):
                mujoco.mj_jacBodyCom(self.m,d,jp,jr,self.head+j);dx=jp[:,self.va]+jp[:,:6]@root;fg=self.m.body_mass[self.head+j]*self.m.opt.gravity
                gradient+=dx[1]*fg[2]-dx[2]*fg[1]
            for ct in sensor['contacts']:
                if known_source(ct,self.source):continue
                mujoco.mj_jac(self.m,d,jp,jr,ct['point'],self.head+ct['material']);dx=jp[:,self.va]+jp[:,:6]@root;force=ct['world_force_N']
                gradient+=dx[1]*force[2]-dx[2]*force[1]
            keep=(np.arange(20)>=p['first_hinge'])&(np.arange(20)<=p['last_hinge']);gradient*=keep
            direction=-(moment-self.baseline)*gradient;error=sensor['joint_ctrl']-sensor['joint_actual'];torque=sensor['joint_torque']
            unavailable=(abs(torque)>=p['saturation_torque_Nm'])&(abs(error)>=np.radians(p['saturation_error_deg']))&(direction*torque>0)
            gradient[unavailable]=0.
            rate=-p['gain_per_s']*(moment-self.baseline)*gradient/(gradient@gradient+p['regularization_Nm2_rad2'])-p['leak_per_s']*self.offset
            rate=np.clip(rate,-np.radians(p['rate_bound_deg_s']),np.radians(p['rate_bound_deg_s']));rate[~keep]=0.
            self.offset=np.clip(self.offset+dt*rate,-np.radians(p['offset_bound_deg']),np.radians(p['offset_bound_deg']))
        fade=1. if first_capture is None else 1.-self.smooth((t-first_capture)/p['post_capture_fade_s'])
        output=fade*self.offset
        info=dict(source_balance_external_X_moment_Nm=moment,source_balance_initial_X_moment_Nm=self.baseline,source_balance_residual_Nm=moment-self.baseline,
            source_balance_gradient_Nm_rad=gradient,source_balance_offset_rad=output.copy(),source_balance_rate_rad_s=rate,source_balance_anchor_material=anchor,source_balance_fade=fade)
        return output,info
    @staticmethod
    def smooth(x):
        x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))
