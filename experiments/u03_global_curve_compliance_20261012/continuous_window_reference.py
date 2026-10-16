"""True geometric arc -> continuous Bishop field -> native alternating integrals.

Material nodes keep64.5mm chords. Their guide-arc supports need not equal64.5mm.
No model state other than once-initialized points/angles enters this module.
"""
from polar_window_reference import PolarWindowReference,LINK
import numpy as np

class ContinuousWindowReference(PolarWindowReference):
    def __init__(self,points,angles):
        super().__init__(points,angles)
        self.gauge_angle=0.
        self.set_nominal_family(0);p,loc=self.body(0,True)
        tangent=-self.spline(loc[0],1);tangent/=np.linalg.norm(tangent)
        normal=np.eye(3)[np.argmin(abs(tangent))];normal-=normal@tangent*tangent;normal/=np.linalg.norm(normal)
        tt=-self.spline(np.linspace(loc[0],self.anchor_arc,600),1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        self.continuous_anchor_normal=self.transport_normal(normal,tt)
        q0,i=self.command(0);field=i['curvature_field'];b=self.joint_bending(i['curvature_bounds_mm'],field,i['material_arc_nodes_mm'])
        q90=-np.where(np.arange(20)%2,-b[:,1],b[:,0])
        weights=np.linalg.lstsq(np.column_stack([q0,q90]),angles,rcond=None)[0]
        self.gauge_angle=float(np.arctan2(weights[1],weights[0]))
        self.spec.update(curvature_mapping='256 exact tangent rotations over TRUE guide arclength; C reconstruction on same arc supports',material_sampling='first backwards64.5mm chord intersections; joint integral support=guide arc between nodes j and j+2; robot lengths not changed',initial_material_gauge_rad=self.gauge_angle,nominal_curve_placement='head point and continuous head tangent of geometric family; no pose repair or translation')

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

    def material_anchor(self,S,locations):
        station=.5*(locations[-1]+locations[-2])-self.start_arc
        tau=self.pitch/(2*np.pi)/(self.radius**2+(self.pitch/(2*np.pi))**2)
        compensation=tau*(station-self.initial_station)
        return self.anchor_arc,self.continuous_anchor_normal.copy(),self.gauge_angle+2*np.pi*S/self.feed_per_cycle-compensation,compensation
