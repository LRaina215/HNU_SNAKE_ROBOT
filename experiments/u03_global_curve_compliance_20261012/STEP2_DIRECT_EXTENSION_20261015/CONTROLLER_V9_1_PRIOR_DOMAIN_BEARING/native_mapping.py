"""Original STEP1 mapping/source functions; target geometry is supplied separately."""
import numpy as np
LINK=.0645

class NativeMapping:
    @staticmethod
    def smooth(x):
        x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))

    def coords(self,p,theta):
        return np.array([p[0],np.linalg.norm(p[1:]-[0,1]),theta])

    @staticmethod
    def polar_derivatives(p,t,acc):
        rad=p[1:]-[0,1];r=np.linalg.norm(rad);er=rad/r;ep=np.array([-er[1],er[0]])
        rd=er@t[1:];td=ep@t[1:]/r
        return np.array([t[0],rd,td]),np.array([acc[0],er@acc[1:]+r*td*td,(ep@acc[1:]-2*rd*td)/r])

    @staticmethod
    def transport_normal(normal,tangents):
        for before,after in zip(tangents[:-1],tangents[1:]):
            cross=np.cross(before,after);sine=np.linalg.norm(cross);cosine=np.clip(before@after,-1,1)
            if sine>1e-12:
                axis=cross/sine;normal=normal*cosine+np.cross(axis,normal)*sine+axis*(axis@normal)*(1-cosine)
        normal-=normal@tangents[-1]*tangents[-1];return normal/np.linalg.norm(normal)

    @staticmethod
    def phase_cells(S):
        x=np.clip((S-5)/4,0,1)
        return min(S,5)+4*(x-2.5*x**4+3*x**5-x**6)

    def feed_distance(self,S):
        x=np.clip((S-6)/2,0,1);integral=2*(2.5*x**4-3*x**5+x**6)+max(0,S-8)
        return self.feed_early*S+(self.feed_late-self.feed_early)*integral

    def source(self,u,phase):
        if not self.supply_enabled:return super().source(u,phase)
        phi=self.phi0-self.beta*u
        return np.column_stack([self.x0+self.a*(u+self.source_chart_shift),self.radius*np.cos(phi),1+self.radius*np.sin(phi)])

    def body(self,S,return_locations=False):
        if not self.supply_enabled:return super().body(S,return_locations)
        tail=self.start_arc+self.tail_u0-self.exit_u+self.feed_distance(S)-self.source_chart_shift;locations=[tail];points=[self.spline(tail)]
        for _ in range(21):
            ds=np.linspace(.1*LINK,2.5*LINK,70);trial=locations[-1]+ds;dist=np.linalg.norm(self.spline(trial)-points[-1],axis=1);ids=np.flatnonzero(dist>=LINK)
            if not len(ids):raise ValueError('no forward chord-length material intersection')
            j=int(ids[0]);hi=trial[j];lo=locations[-1] if j==0 else trial[j-1]
            for _ in range(12):
                mid=(hi+lo)/2
                if np.linalg.norm(self.spline(mid)-points[-1])>=LINK:hi=mid
                else:lo=mid
            locations.append((hi+lo)/2);points.append(self.spline(locations[-1]))
        p=np.asarray(points)[::-1];loc=np.asarray(locations)[::-1];return (p,loc) if return_locations else p

    def material_anchor(self,S,locations):
        if not self.supply_enabled:return super().material_anchor(S,locations)
        anchor=self.start_arc+self.anchor_u0+self.feed_distance(S)-self.source_chart_shift-self.exit_u;delta=self.spatial_phase-self.beta*self.feed_distance(S);n=self.supply_anchor_normal.copy();co,sn=np.cos(delta),np.sin(delta);n[1:]=[co*n[1]-sn*n[2],sn*n[1]+co*n[2]]
        return anchor,n,self.gauge_angle+2*np.pi*S/self.feed_per_cycle,0.

    @staticmethod
    def joint_bending(bounds,field,nodes):
        lo=nodes[:-2];hi=nodes[2:]
        lengths=np.maximum(0,np.minimum(hi[:,None],bounds[None,1:])-np.maximum(lo[:,None],bounds[None,:-1]))
        return lengths@field

    def command(self,S):
        self.set_nominal_family(S);p,loc=self.body(S,True);L=loc[0]-loc[-1]
        arc=np.linspace(0,L,257);tt=-self.spline(loc[0]-arc,1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        anchor,anchor_normal,angle,compensation=self.material_anchor(S,loc)
        path=np.linspace(anchor,loc[0],max(2,int((loc[0]-anchor)/.004)+1));srcT=-self.spline(path,1);srcT/=np.linalg.norm(srcT,axis=1)[:,None]
        normal=self.transport_normal(anchor_normal,srcT)
        station=.5*(loc[-1]+loc[-2])-self.start_arc
        tau=self.pitch/(2*np.pi)/(self.radius**2+(self.pitch/(2*np.pi))**2)
        chi=S/self.feed_per_cycle
        normal=normal*np.cos(angle)-np.cross(tt[0],normal)*np.sin(angle)
        initial_frame=np.column_stack([tt[0],normal,np.cross(tt[0],normal)])
        fields=[];curve=[p[0].copy()];normals=[normal.copy()]
        for j,(before,after) in enumerate(zip(tt[:-1],tt[1:])):
            cross=np.cross(before,after);sn=np.linalg.norm(cross);co=np.clip(before@after,-1,1);theta=np.arctan2(sn,co);span=arc[j+1]-arc[j]
            if sn>1e-12:
                axis=cross/sn;rotvec=axis*theta;derivative=np.cross(rotvec/span,before)
                second=np.cross(before,normal);fields.append(np.array([derivative@normal,derivative@second])/1000)
                delta=span*(np.sin(theta)/theta*before+(1-np.cos(theta))/theta*np.cross(axis,before))
                normal=normal*co+np.cross(axis,normal)*sn+axis*(axis@normal)*(1-co)
            else:fields.append(np.zeros(2));delta=span*before
            curve.append(curve[-1]+delta);normals.append(normal.copy())
        field=np.array(fields);bounds=arc*1000;material=(loc[0]-loc)*1000;b=self.joint_bending(bounds,field,material)
        q=-np.where(np.arange(20)%2,b[:,0],b[:,1])
        cc=np.asarray(curve);C=np.column_stack([np.interp(material, bounds,cc[:,j]) for j in range(3)])
        canonical=np.eye(3)[np.argmin(abs(tt[0]))];canonical-=canonical@tt[0]*tt[0];canonical/=np.linalg.norm(canonical)
        rotation=np.column_stack([canonical,np.cross(tt[0],canonical)]).T@initial_frame[:,1:]
        return q,dict(S=S,chi=chi,roll_phase=angle,nominal_pipe_phase=np.zeros(2),local_material_phase=np.zeros(256),curvature_field=field,curvature_bounds_mm=bounds,material_arc_nodes_mm=material,material_rotation=rotation,reference_points=p,reference_locations=loc,reference_frame=initial_frame,nominal_curve_points=C,nominal_curve_dense=cc,source_reference=float(np.mean(loc<self.start_arc)),target_reference=float(np.mean(loc>self.target_arc)),connector_arc_m=self.connector_arc,nominal_shift_cells=S,source_station_transport_rad=compensation,uncoil_fraction=self.grow,receiving_recovery_fraction=self.recover,true_guide_body_arc_m=L,guide_to_C_max_mm=float(np.max(np.linalg.norm(C-p,axis=1)))*1000)

