"""One bounded whole-bridge shape check before deciding whether C03 can run."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco,json
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.cmu_bridge_candidate import BridgeCandidate
from snake_control.cmu_window_gait import calibrate,wave,cylinder
from snake_control.suspended_supply_transfer import nodes
from scipy.spatial.transform import Rotation
from validation.cmu_window_experiment import BASE,IMODEL
from validation.suspended_supply_experiment import write,default

class AxisBridge(BridgeCandidate):
    def start(self,x):
        y=np.zeros(8);y[4:6]=x[6:8];return super().start(y)
    def joint_wave(self,x,s):
        h=self.front;b=self.bend;k=np.arange(20);q=wave(self.A,self.w,self.ph);u=self.path_u(s);loose=np.sin(np.pi*u)
        desired=q.copy();desired[:h]=wave(self.A*(1-.4*loose),self.w,self.ph+x[0]*u)[:h]
        for parity in [0,1]:
            idx=np.arange(h,b);idx=idx[idx%2==parity];amp=((1-u)*self.A+u*x[1+parity])*(1-.35*loose);wb=(1-u)*self.w+u*x[3];pb=(1-u)*(self.ph+self.w*h+parity*np.pi/2)+u*x[4+parity];desired[idx]=amp*np.sin(wb*(idx-h)+pb)
        if self.path_mode=='sequential':
            weights=np.clip((s*(b+2)-k)/2,0,1);weights=weights*weights*(3-2*weights);desired[:b]=q[:b]+weights[:b]*(desired[:b]-q[:b])
        return desired
    def path_u(self,s):return s
    def endpoint_residual(self,x):
        self.start(x);out=[];h=self.front
        for s in [0,1]:
            r=self.pose(x,s);c=r['centers'];out.extend(np.maximum(r['pipe_depth']-.0012,0)/.003);out.extend(np.maximum(r['self_depth']-.0005,0)/.003)
            if s==0:out.extend(np.maximum(c[:,0]-1.085,0)/.01);out.append(r['target_contact_count']*2.)
            else:
                cc=c[:h];rr=np.linalg.norm(cc[:,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(cc[:,2]-1,cc[:,0]-1.1));dy=np.diff(cc[:,1]);dp=np.diff(phi);out.extend((rr-.088)/.003);out.extend((dy+.18/(2*np.pi)*dp)/.003);out.extend(np.maximum(.3-dp,0)/.1);out.extend(np.maximum(dy+.002,0)/.01);out.extend(np.maximum(.15-cc[:,1],0)/.01)
                entry=np.array([1.1+.088*np.cos(5*np.pi/4),.18,1+.088*np.sin(5*np.pi/4)]);out.extend((cc[-1]-entry)/.008)
        return np.asarray(out)

class FrameBridge(AxisBridge):
    def target_frame(self,phase):
        d=mujoco.MjData(self.m);d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[self.qa]=wave(self.A,self.w,phase);mujoco.mj_fwdPosition(self.m,d);p=nodes(d);cs=cylinder(p);R=Rotation.align_vectors([[0,1.,0]],[cs['axis']])[0].as_matrix();c=((p[:-1]+p[1:])/2-cs['origin'])@R.T;angle=np.arctan2(c[self.front-1,2],c[self.front-1,0])-5*np.pi/4;R=Rotation.from_rotvec([0,angle,0]).as_matrix()@R
        pp=(p-cs['origin'])@R.T;cc=(pp[:-1]+pp[1:])/2;trans=np.array([1.1,.18-cc[self.front-1,1],1])-R@cs['origin'];return R,trans
    def endpoint_residual(self,x):
        self.start(x);out=[]
        for s in [0,1]:
            r=self.pose(x,s);c=r['centers'];out.extend(np.maximum(r['pipe_depth']-.0012,0)/.003);out.extend(np.maximum(r['self_depth']-.0005,0)/.003)
            if s==0:out.extend(np.maximum(c[:,0]-1.085,0)/.01);out.append(r['target_contact_count']*2.)
            else:
                R,trans=self.target_frame(self.ph+x[0]);out.extend((r['qpos'][:3]-trans)/.006);Ractual=Rotation.from_quat(r['qpos'][3:7][[1,2,3,0]]).as_matrix();out.extend(Rotation.from_matrix(Ractual@R.T).as_rotvec()/.04)
        return np.asarray(out)

def check(frame_constraints=True):
    out=BASE/'C03_PREPARE';m=mujoco.MjModel.from_xml_path(str(IMODEL));A,w,ph,_,_=calibrate(m);c=(FrameBridge if frame_constraints else AxisBridge)(m,A,w,ph);c.path_mode='simultaneous';stem='frame_bridge' if frame_constraints else 'axis_bridge';bounds=([-np.pi,.1,.1,-1,-np.pi,-np.pi,.70,-.8],[np.pi,np.radians(88),np.radians(88),1,np.pi,np.pi,.95,.8]);fits=[]
    for sign in [-1,1]:
        x0=[sign*np.pi/2,1.,1.,.1,.4,.4+np.pi/2,.82,0];fit=least_squares(c.endpoint_residual,x0,bounds=bounds,max_nfev=160,ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=1e-4);fits.append(fit);print('endpoint',sign,'cost',fit.cost,'nfev',fit.nfev,flush=True)
    fit=min(fits,key=lambda f:f.cost);x=fit.x;initial=c.start(x);allpaths=[]
    for mode in ['simultaneous','sequential']:
        c.path_mode=mode;records=[]
        for s in np.linspace(0,1,201):
            r=c.pose(x,s);cc=r['centers'][:c.front];rr=np.linalg.norm(cc[:,[0,2]]-[1.1,1],axis=1);phi=np.unwrap(np.arctan2(cc[:,2]-1,cc[:,0]-1.1));records.append(dict(s=s,q_rad=r['q'],qpos=r['qpos'],points_mm=r['points']*1000,pipe_depth_mm=float(r['pipe_depth'].max()*1000),self_depth_mm=float(r['self_depth'].max()*1000),target_contact_count=r['target_contact_count'],front_radius_mm=rr*1000,front_wrap_deg=float(np.degrees(np.ptp(phi))),unsupported_gravity_torque_nm=r['unsupported_gravity_torque_nm']))
        final=records[-1];rr=final['front_radius_mm'];hook=bool(np.all((rr>75)&(rr<105)) and final['front_wrap_deg']>180 and final['points_mm'][:c.front+1,1].min()>150);pd=max(r['pipe_depth_mm'] for r in records);sd=max(r['self_depth_mm'] for r in records);entrygap=np.linalg.norm(final['points_mm'][c.front]-records[0]['points_mm'][c.bend]);margin=774-entrygap-.75*np.hypot(2*np.pi*88,180)
        feasible=bool(hook and pd<1.5 and sd<.5 and records[0]['target_contact_count']==0 and margin>0)
        result=dict(path_mode=mode,parameters=x,global_parameter_names=['head_phase_offset','bridge_even_amplitude','bridge_odd_amplitude','bridge_spatial','bridge_even_phase','bridge_odd_phase','source_exit_x','source_polar_rotation'],release_rods=12,release_length_mm=774,posterior_source_rods=9,front_target_rods=7,bridge_rods=5,actual_reference_entry_gap_mm=entrygap,exact270deg_wrap_length_mm=.75*np.hypot(2*np.pi*88,180),length_margin_mm=margin,cost=float(fit.cost),endpoint_hook_shape=hook,maximum_reference_pipe_depth_mm=pd,maximum_reference_self_depth_mm=sd,initial_target_contact_count=records[0]['target_contact_count'],final_front_radius_mm=rr,final_front_wrap_deg=final['front_wrap_deg'],geometrically_executable=feasible,physical_steps=0,classification='C03 offline two-axis bridge-wave adaptation check; no independent per-hinge fitting; not paper baseline',records=records);allpaths.append(result);print(json.dumps({k:v for k,v in result.items() if k!='records'},default=default),flush=True)
    write(out/(stem+'_shape_check.json'),allpaths);best=min(allpaths,key=lambda r:(not r['geometrically_executable'],not r['endpoint_hook_shape'],r['maximum_reference_pipe_depth_mm']));write(out/(stem+'_candidate.json'),best)
    fig=plt.figure(figsize=(17,10))
    for i,j in enumerate([0,40,80,120,160,200]):
        r=best['records'][j];p=r['points_mm']/1000;ax=fig.add_subplot(2,3,i+1,projection='3d');ax.plot(*p[:8].T,'o-',c='green',ms=3,label='Target-wave7');ax.plot(*p[7:13].T,'o-',c='orange',ms=3,label='Two-axis bridge-wave5');ax.plot(*p[12:].T,'o-',c='navy',ms=3,label='Source9');ax.scatter(*p[0],c='red',label='Head');ax.plot([.4,1.1],[0,0],[1,1],lw=9,c='gray',alpha=.3);ax.plot([1.1,1.1],[0,.6],[1,1],lw=9,c='gray',alpha=.3);ax.set(title=f'PRIVATE FK {best["path_mode"]} s={r["s"]:.2f}, pipe={r["pipe_depth_mm"]:.1f}mm',xlabel='X(m)',ylabel='Y(m)',zlabel='Z UP(m)');ax.view_init(27,140);ax.legend(fontsize=7)
    fig.suptitle('C03 '+('frame-constrained' if frame_constraints else 'axis-bridge')+' hinge geometry in private FK / no dynamics executed');fig.tight_layout();fig.savefig(out/(stem+'_complete_joint_shape.png'),dpi=150);plt.close(fig)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--axis-only',action='store_true');args=ap.parse_args();check(frame_constraints=not args.axis_only)
