import numpy as np
from physics import mujoco,to_world_points,to_world_vectors
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


class SourceMode:
    def observe_all(self,m,d,observation,roles,previous,dt,shape_enabled):
        if shape_enabled:self.observe(observation['centers'],observation['forces'][1],previous,dt)
        else:
            self.delta=0.;self.observation.update(compliance_delta_m=0.,compliance_rate_m_s=0.,compliance_amplitude_saturated=False,compliance_rate_saturated=False)
        index=roles['source_loaded_material'];force=roles['source_late_filtered_N']
        active=shape_enabled and (roles['front_confirmed'] or getattr(self,'receiving_bearing_valid',False)) and index>=17 and force>.5
        settings=getattr(self,'gravity_contact_C',None);bound=settings.get('source_relief_curvature_bound_per_m',.4) if settings else .4;rate_bound=settings.get('source_relief_curvature_rate_per_m_s',.08) if settings else .08
        gradient=0.;desired=0.
        if active:
            n=observation['contact_normal'][0,index];point=observation['contact_point'][0,index]
            if np.linalg.norm(n)>.5:
                bounds=previous['curvature_bounds_mm'];nodes=previous['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:])
                v=np.clip((mid-nodes[15])/(nodes[-1]-nodes[15]),0,1);weight=np.sin(np.pi*v)**2;weight[(bounds[:-1]<nodes[15])|(bounds[1:]>nodes[-1])]=0.
                lengths=np.maximum(0,np.minimum(nodes[2:,None],bounds[None,1:])-np.maximum(nodes[:-2,None],bounds[None,:-1]));span=lengths@weight/1000
                mapping=np.zeros((20,2));odd=np.arange(20)%2==1;mapping[odd,0]=-span[odd];mapping[~odd,1]=-span[~odd]
                jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,to_world_points(point),m.body('head').id+index)
                columns=m.jnt_dofadr[m.actuator_trnid[:,0]];raw_g=to_world_vectors(n)@jp[:,columns]@mapping;g=raw_g.copy()
                error=observation['joint_ctrl']-observation['joint_actual'];torque=observation['joint_torque']
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
                        settings=getattr(self,'gravity_contact_C',None);target=settings['source_relief_load_target_N'] if settings and getattr(self,'receiving_bearing_valid',False) else 12.
                        gain=settings.get('source_relief_force_gain_per_m_N',.015) if settings else .015
                        depth_gain=settings.get('source_relief_depth_gain_per_m',.08) if settings else .08
                        desired=float(np.clip(gain*max(0.,force-target)+depth_gain*max(0.,(roles['source_late_depth_m']-.0008)/.001),0,bound))
        before=self.source_gamma;rate=np.clip((desired-before)/(.4 if active else .8),-rate_bound,rate_bound)
        self.source_gamma=float(np.clip(before+dt*rate,0,bound))
        if self.source_gamma<1e-6 and desired==0:self.source_gamma=0.
        self.source_gradient=gradient;self.source_requested=desired
        return dict(source_gamma_per_m=self.source_gamma,source_gamma_rate_per_m_s=(self.source_gamma-before)/dt,
            source_curve_direction=self.source_direction.copy(),source_outward_gradient_m2=gradient,
            source_gamma_requested_per_m=desired,source_mode_observed_material=index,source_mode_active=active)


