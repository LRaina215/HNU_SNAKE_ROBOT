"""CMU2025 equations8/9/12, adapted to native20 hinges, head-index first."""
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from snake_control.suspended_supply_transfer import nodes,LINK,DT,LIMIT

PIPE_D=.113;MODULE_D=.058;CENTRE_R=(PIPE_D+MODULE_D)/2
EMPIRICAL_BEND=int(np.floor(3*np.pi/4*(PIPE_D+MODULE_D)/LINK))

def wave(amplitude,spatial,phase):
    k=np.arange(20)
    # J1/J3/... have nativeX axes; J2/J4/... native-Y. At zero shape
    # the alternating180deg child frames make their world axes -headX/-headY.
    return amplitude*np.sin(spatial*k+phase+(k%2)*np.pi/2)

def cylinder(p):
    c=(p[:-1]+p[1:])/2;centre=c.mean(0);_,_,vh=np.linalg.svd(c-centre);a=vh[0]
    if a@(c[0]-c[-1])<0:a=-a
    u=np.eye(3)[np.argmin(abs(a))];u-=u@a*a;u/=np.linalg.norm(u);v=np.cross(a,u)
    def unpack(x):
        axis=a+x[0]*u+x[1]*v;axis/=np.linalg.norm(axis);origin=centre+x[2]*u+x[3]*v;return axis,origin
    def residual(x):
        axis,origin=unpack(x);r=c-origin;return np.sqrt(np.maximum(0,(r*r).sum(1)-(r@axis)**2))-x[4]
    r=least_squares(residual,[0,0,0,0,CENTRE_R],bounds=([-.5,-.5,-.2,-.2,.04],[.5,.5,.2,.2,.2]),max_nfev=100)
    axis,origin=unpack(r.x);uu=u-u@axis*axis;uu/=np.linalg.norm(uu);vv=np.cross(axis,uu);rr=c-origin;phi=np.unwrap(np.arctan2(rr@vv,rr@uu));coef=np.polyfit(phi,rr@axis,1)
    return dict(axis=axis,origin=origin,radius=float(r.x[4]),pitch=float(abs(coef[0])*2*np.pi),radial_rms=float(np.sqrt(np.mean(r.fun*r.fun))),signed_pitch=float(coef[0]*2*np.pi))

def calibrate(model,phase=-.9,radius=.088):
    d=mujoco.MjData(model);qa=model.jnt_qposadr[model.actuator_trnid[:,0]]
    def evaluate(x):
        d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=wave(x[0],x[1],phase);mujoco.mj_fwdPosition(model,d);p=nodes(d);s=cylinder(p);return p,s
    def residual(x):
        _,s=evaluate(x);return [(s['radius']-radius)/.003,(s['pitch']-.18)/.01]
    result=least_squares(residual,[1.23,.196],bounds=([.8,.08],[float(LIMIT),.4]),max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    p,s=evaluate(result.x);return float(result.x[0]),float(result.x[1]),phase,p,s

def placed_pose(model,amplitude,spatial,source_phase,bend,seam_x):
    d=mujoco.MjData(model);qa=model.jnt_qposadr[model.actuator_trnid[:,0]];d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=wave(amplitude,spatial,source_phase);mujoco.mj_fwdPosition(model,d);p=nodes(d);s=cylinder(p)
    align=Rotation.align_vectors([[1.,0,0]],[s['axis']])[0].as_matrix();rr=(p-s['origin'])@align.T;c=(rr[:-1]+rr[1:])/2
    angle=-np.arctan2(c[bend,2],c[bend,1]);rot=Rotation.from_rotvec([angle,0,0]).as_matrix()@align
    shifted=(p-s['origin'])@rot.T;translation=np.array([seam_x-shifted[bend,0],0,1])-rot@s['origin']
    d.qpos[:3]=translation;d.qpos[3:7]=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]];mujoco.mj_fwdPosition(model,d)
    return d.qpos.copy(),nodes(d),s

class CMUWindowControl:
    def __init__(self,model,data,amplitude,spatial,source_phase,bend,frequency=.02,klow=.4,slope=5):
        self.m=model;self.qa=model.jnt_qposadr[model.actuator_trnid[:,0]];self.A=amplitude;self.ws=spatial;self.nu=int(np.sign(amplitude*spatial));self.delta=-self.nu*np.pi/2
        self.phase0=source_phase-self.delta;self.bend=bend;self.f=frequency;self.omega=-2*np.pi*frequency
        self.window_rate=np.pi*MODULE_D/LINK*abs(frequency);self.k_low=klow;self.slope=slope;self.last=data.ctrl.copy();self.kp=np.full(20,127.);self.received=set();self.exited=set();self.failure=None;self.shift=0.;self.phase=self.phase0
        self.fk=mujoco.MjData(model);self.head=model.body('head').id

    def forward_reference(self,data,q,anchor):
        # Private FK only. Rigidly register one carrying module's complete frame;
        # this never changes the physical state and never fits a world curve.
        d=self.fk;d.qpos[:]=data.qpos;d.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,d)
        body=self.head+anchor;rot=data.xmat[body].reshape(3,3)@d.xmat[body].reshape(3,3).T
        trans=data.xpos[body]-rot@d.xpos[body]
        root=rot@d.xmat[self.head].reshape(3,3)
        d.qpos[:3]=rot@d.qpos[:3]+trans;d.qpos[3:7]=Rotation.from_matrix(root).as_quat()[[3,0,1,2]];mujoco.mj_fwdPosition(self.m,d)
        return nodes(d)

    def reference(self,t):
        nt=self.window_rate*t;k=np.arange(20);phase=self.phase0+self.omega*t;head=k<=nt
        L=self.k_low+(1-self.k_low)/(1+np.exp(np.clip(-self.slope*(nt-self.bend),-60,60)))
        raw=wave(self.A,self.ws,phase+0.)
        tail=wave(self.A,self.ws,phase+self.delta);raw=np.where(head,raw,tail);raw[(k<self.bend)&head]*=L
        return raw,nt,L,head,phase

    def command(self,data,t,forces,source_any):
        raw,nt,L,head,phase=self.reference(t);before=self.last.copy();target=np.clip(raw,np.maximum(-LIMIT,before-np.radians(20)*DT),np.minimum(LIMIT,before+np.radians(20)*DT));self.last=target.copy();self.phase=phase;self.shift=min(nt,20)*LINK
        p=nodes(data);c=(p[:-1]+p[1:])/2;r=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1);self.received=set(np.flatnonzero((r>.065)&(r<.125)&(c[:,1]>.12)&(forces[1]>.1)));sr=np.linalg.norm(c[:,[1,2]]-[0,1],axis=1);self.exited=set(np.flatnonzero((sr>.13)&(~source_any)&(forces[0]<.1)))
        bearing=np.flatnonzero(source_any & (np.arange(21)>=self.bend));anchor=int(bearing[0]) if len(bearing) else 18
        ref=self.forward_reference(data,target,anchor)
        return target,dict(stage='cmu_head_loosen' if nt<self.bend else 'cmu_window_tailward' if nt<20 else 'cmu_all_head_window',shift_mm=self.shift*1000,roll_phase_rad=phase,window_transition_modules=nt,window_rate_modules_s=self.window_rate,temporal_phase_rate_rad_s=self.omega,cycles_per_second=self.f,loosening_factor=L,bend_actuators=self.bend,head_window_mask=head,raw_wave_rad=raw,maximum_dispatch_step_deg=float(np.degrees(abs(target-before)).max()),tracking_deg=float(np.degrees(abs(target-data.qpos[self.qa])).max()),kp_nm_per_rad=self.kp.copy(),actual_received_modules=sorted(self.received),actual_source_exited_modules=sorted(self.exited),actual_new_retained_modules=sorted(self.received),reference_kind='direct sine joint waves; no world-curve inverse fit',reference_points_mm=ref*1000,reference_anchor_module=anchor,yamada_compliance_enabled=False,physical_root_assigned=False)
