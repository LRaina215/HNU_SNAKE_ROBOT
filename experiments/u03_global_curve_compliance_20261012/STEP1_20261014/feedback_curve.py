"""Receiving radial compliance plus one late source-exit curvature mode."""
import numpy as np
from local_curve_compliance import LocalCurveCompliance
from physics import mujoco


def reconstruct(field,bounds,frame,head,nodes):
    T=frame[:,0].copy();N=frame[:,1].copy();point=head.copy();curve=[point.copy()]
    for k,span in zip(field,np.diff(bounds)/1000):
        B=np.cross(T,N);omega=k[0]*1000*B-k[1]*1000*N;rate=np.linalg.norm(omega)
        if rate>1e-12:
            axis=omega/rate;angle=rate*span;co=np.cos(angle);sn=np.sin(angle)
            point+=span*(sn/angle*T+(1-co)/angle*np.cross(axis,T))
            T=T*co+np.cross(axis,T)*sn+axis*(axis@T)*(1-co)
            N=N*co+np.cross(axis,N)*sn+axis*(axis@N)*(1-co)
        else:point+=span*T
        curve.append(point.copy())
    dense=np.asarray(curve);points=np.column_stack([np.interp(nodes,bounds,dense[:,j]) for j in range(3)])
    return dense,points


class FeedbackCurve(LocalCurveCompliance):
    def __init__(self,points,angles):
        self.source_gamma=0.;self.source_direction=np.array([0.,1.]);self.source_gradient=0.;self.source_requested=0.
        super().__init__(points,angles)
        self.spec.update(step1_source_mode='bounded curvature mode after material node15, world outward direction from actual late-source contact and read-only material Jacobian; first14 joint integrals protected',
            source_gamma_max_per_m=.4,source_gamma_rate_per_m_s=.08,source_force_reference_N=12,
            source_direction_filter_s=.5,source_feedback_requires_confirmed_front=True)

    def observe_all(self,m,d,truth,roles,previous,dt,shape_enabled):
        if shape_enabled:self.observe(truth['centers'],truth['forces'][1],previous,dt)
        else:
            self.delta=0.;self.observation.update(compliance_delta_m=0.,compliance_rate_m_s=0.,compliance_amplitude_saturated=False,compliance_rate_saturated=False)
        index=roles['source_loaded_material'];force=roles['source_late_filtered_N']
        active=shape_enabled and roles['front_confirmed'] and index>=17 and force>.5
        gradient=0.;desired=0.
        if active:
            n=truth['contact_normal'][0,index];point=truth['contact_point'][0,index]
            if np.linalg.norm(n)>.5:
                bounds=previous['curvature_bounds_mm'];nodes=previous['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:])
                v=np.clip((mid-nodes[15])/(nodes[-1]-nodes[15]),0,1);weight=np.sin(np.pi*v)**2;weight[(bounds[:-1]<nodes[15])|(bounds[1:]>nodes[-1])]=0.
                lengths=np.maximum(0,np.minimum(nodes[2:,None],bounds[None,1:])-np.maximum(nodes[:-2,None],bounds[None,:-1]));span=lengths@weight/1000
                mapping=np.zeros((20,2));odd=np.arange(20)%2==1;mapping[odd,0]=-span[odd];mapping[~odd,1]=-span[~odd]
                jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,point,m.body('head').id+index)
                columns=m.jnt_dofadr[m.actuator_trnid[:,0]];raw_g=n@jp[:,columns]@mapping;g=raw_g.copy()
                error=truth['joint_ctrl']-truth['joint_actual'];torque=truth['joint_torque']
                allowed=[]
                for component in range(2):
                    affected=abs(mapping[:,component])>1e-12
                    positive=np.any(affected&(torque>2.35)&(error>np.radians(1)))
                    negative=np.any(affected&(torque<-2.35)&(error<-np.radians(1)))
                    allowed.append((positive,negative))
                    if positive:g[component]=max(0.,g[component])
                    if negative:g[component]=min(0.,g[component])
                gradient=float(np.linalg.norm(g))
                if gradient>1e-7:
                    direction=g/gradient;alpha=1-np.exp(-dt/.5);self.source_direction+=alpha*(direction-self.source_direction)
                    for component,(positive,negative) in enumerate(allowed):
                        if positive:self.source_direction[component]=max(0.,self.source_direction[component])
                        if negative:self.source_direction[component]=min(0.,self.source_direction[component])
                    self.source_direction/=max(np.linalg.norm(self.source_direction),1e-12)
                    if raw_g@self.source_direction>0:
                        desired=float(np.clip(.015*max(0.,force-12)+.08*max(0.,(roles['source_late_depth_m']-.0008)/.001),0,.4))
        before=self.source_gamma;rate=np.clip((desired-before)/(.4 if active else .8),-.08,.08)
        self.source_gamma=float(np.clip(before+dt*rate,0,.4))
        if self.source_gamma<1e-6 and desired==0:self.source_gamma=0.
        self.source_gradient=gradient;self.source_requested=desired
        return dict(source_gamma_per_m=self.source_gamma,source_gamma_rate_per_m_s=(self.source_gamma-before)/dt,
            source_curve_direction=self.source_direction.copy(),source_outward_gradient_m2=gradient,
            source_gamma_requested_per_m=desired,source_mode_observed_material=index,source_mode_active=active)

    def command(self,S):
        q,info=super().command(S)
        if not self.compliance_ready:return q,info
        info.update(receiving_curve_points=info['reference_points'].copy(),source_gamma_per_m=self.source_gamma,
            source_curve_direction=self.source_direction.copy())
        if self.source_gamma>0:
            field=info['curvature_field'].copy();bounds=info['curvature_bounds_mm'];nodes=info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:])
            v=np.clip((mid-nodes[15])/(nodes[-1]-nodes[15]),0,1);weight=np.sin(np.pi*v)**2;weight[(bounds[:-1]<nodes[15])|(bounds[1:]>nodes[-1])]=0.
            field+=self.source_gamma*weight[:,None]*self.source_direction[None,:]/1000
            b=self.joint_bending(bounds,field,nodes);q=-np.where(np.arange(20)%2,b[:,0],b[:,1])
            dense,C=reconstruct(field,bounds,info['reference_frame'],info['reference_points'][0],nodes)
            info.update(curvature_field=field,nominal_curve_dense=dense,nominal_curve_points=C,
                reference_points=C,local_curve_chord_error_mm=float(np.max(abs(np.linalg.norm(np.diff(C,axis=0),axis=1)-.0645)))*1000)
        else:info['local_curve_chord_error_mm']=float(np.max(abs(np.linalg.norm(np.diff(info['nominal_curve_points'],axis=0),axis=1)-.0645)))*1000
        return q,info
