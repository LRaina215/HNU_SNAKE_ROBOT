"""Geometric global guide -> common material window -> Bishop integral angles.

No direct joint wave and no runtime observations enter this module. Distances m.
"""
from pathlib import Path
import sys,json
import math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from geometry_backbone import SpatialBishopBackbone

LINK=.0645

class GlobalCurveReference:
    def __init__(self,initial_points,initial_angles,radius=.0895,pitch=.183,bridge_depth=.76,roll_sign=1.,feed_turns=1.,moving_family=False,anchor_frame=False,bezier_bridge=False,local_material=False,smooth_source=False,discrete_transport=False,axial_family=False,release_curve=False):
        p=np.asarray(initial_points);self.radius=radius;self.pitch=pitch
        # Capture the original geometric source once. It is not a series of FK poses.
        # u increases from tail to head; the initial head is u=0.
        us=-np.arange(22)*LINK;source=CubicSpline(us[::-1],p[::-1],axis=0)
        polar=np.unwrap(np.arctan2(p[::-1,2]-1,p[::-1,1]));coef=np.polyfit(p[::-1,0],polar,1)
        tail=np.arange(-3.,us[-1]-.002,.002);a=pitch/np.sqrt((2*np.pi*radius)**2+pitch**2)
        x=p[-1,0]+a*(tail-us[-1]);phi=np.arctan2(p[-1,2]-1,p[-1,1])+coef[0]*(x-p[-1,0])
        extended=np.column_stack([x,radius*np.cos(phi),1+radius*np.sin(phi)])
        # Independently planned known-pipe corridor, with an upward side entry.
        entry=np.array([1.1+radius,.27,1.]);lower=np.array([1.25,.14,bridge_depth]);t0=(p[0]-p[1])/LINK
        target_t=np.array([0.,a,np.sqrt(1-a*a)])
        controls=np.vstack([p[0],p[0]+.08*t0,lower,entry-.08*target_t,entry])
        distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(controls,axis=0),axis=1))]
        bridge=CubicSpline(distances,controls,axis=0,bc_type=((1,t0),(1,target_t)))
        bu=np.arange(.002,distances[-1]+.001,.002);bp=bridge(bu)
        target_len=np.arange(.002,4.,.002);phi_t=target_len/np.sqrt(radius*radius+(pitch/(2*np.pi))**2)
        tp=np.column_stack([1.1+radius*np.cos(phi_t),.27+a*target_len,1+radius*np.sin(phi_t)])
        su=np.linspace(us[-1],0,680);sp=source(su)
        if smooth_source:
            beta=1/np.sqrt(radius*radius+(pitch/(2*np.pi))**2);phi0=np.arctan2(p[0,2]-1,p[0,1])
            def smooth_helix(values):
                angle=phi0-beta*values;return np.column_stack([p[0,0]+a*values,radius*np.cos(angle),1+radius*np.sin(angle)])
            extended=smooth_helix(tail);sp=smooth_helix(su);p0=sp[-1];t0=np.array([a,radius*beta*np.sin(phi0),-radius*beta*np.cos(phi0)])
        else:p0=p[0]
        raw=np.vstack([extended,sp,bp,tp]);u=np.r_[tail,su,bu,distances[-1]+target_len]
        self.anchor_frame=anchor_frame;self.bezier_bridge=bezier_bridge;self.local_material=local_material;self.discrete_transport=discrete_transport;self.axial_family=axial_family;self.release_curve=release_curve
        if bezier_bridge:
            controls=np.vstack([p0,p0+.05*t0,[1.24,.04,bridge_depth],[1.24,.20,bridge_depth],entry-.05*target_t,entry])
            bu=np.arange(.002,.70,.002);fraction=bu/.70
            bp=sum(math.comb(5,k)*(1-fraction[:,None])**(5-k)*fraction[:,None]**k*controls[k] for k in range(6))
            raw=np.vstack([extended,sp,bp,tp]);u=np.r_[tail,su,bu,.70+target_len];distances=np.linspace(0,.70,6)
        self.raw=CubicSpline(u,raw,axis=0)
        self.construction_u=u.copy();self.construction_xyz=raw.copy();self.bridge_parameter_end=distances[-1];self.bridge_controls=controls.copy();self.bridge_start_t=t0.copy();self.bridge_end_t=target_t.copy();self.moving_family=moving_family
        grid=np.arange(u[0],u[-1],.001);xyz=self.raw(grid);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.start_arc=float(np.interp(0,grid,arc));self.target_arc=float(np.interp(distances[-1],grid,arc))
        self.global_spline=self.spline;self.global_start_arc=self.start_arc;self.global_target_arc=self.target_arc
        self.turn_arc=np.sqrt((2*np.pi*radius)**2+pitch**2);self.feed_per_cycle=feed_turns*self.turn_arc;self.roll_sign=roll_sign
        initial=self.body(0.);bone=SpatialBishopBackbone(initial*1000,np.full(21,LINK*1000));_,rot=bone.material_field(initial_angles)
        material=bone.start_frame[:,1:]@rot
        # Continuous Bishop transport along the nominal head path. Frenet normal
        # can flip at the source/bridge inflection even while tangent is smooth.
        travel=np.r_[self.start_arc,np.arange(self.start_arc+.001,arc[-1],.001)]
        tangent=-self.spline(travel,1);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        normal=material[:,0].copy();normal-=normal@tangent[0]*tangent[0];normal/=np.linalg.norm(normal);transported=[normal.copy()]
        for before,after in zip(tangent[:-1],tangent[1:]):
            cross=np.cross(before,after);sine=np.linalg.norm(cross);angle=np.arctan2(sine,np.clip(before@after,-1,1))
            if sine>1e-12:normal=Rotation.from_rotvec(cross*angle/sine).apply(normal)
            normal-=normal@after*after;normal/=np.linalg.norm(normal);transported.append(normal.copy())
        self.normal_spline=CubicSpline(travel,np.asarray(transported),axis=0)
        self.transported_normal=transported[0].copy();self.last_reference_tangent=tangent[0].copy()
        self.anchor_arc=self.start_arc-1.48
        backward=np.linspace(self.start_arc,self.anchor_arc,500);tt=-self.spline(backward,1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        normal=transported[0].copy();normal-=normal@tt[0]*tt[0];normal/=np.linalg.norm(normal)
        self.anchor_normal=self.transport_normal(normal,tt)
        self.torsion_compensation=(pitch/(2*np.pi))/(radius*radius+(pitch/(2*np.pi))**2)
        self.initial_error_m=float(np.max(np.linalg.norm(initial-p,axis=1)))
        self.spec=dict(version=2,representation='fixed world source curve + lower corridor + upward target helix',source_initial_points_m=p.tolist(),corridor_controls_m=controls.tolist(),radius_m=radius,pitch_m=pitch,source_chart_rate_rad_per_x=float(coef[0]),target_chart_rate_rad_per_y=2*np.pi/pitch,initial_head_arc_m=self.start_arc,target_begin_arc_m=self.target_arc,turn_arc_m=self.turn_arc,feed_per_rolling_cycle_m=self.feed_per_cycle,material_window='same head arc = initial_head_arc+S; each next node is first backward chord-length intersection',material_frame='continuous Bishop transport from fixed initial gauge; roll angle 2pi*chi-torsion_compensation*S',torsion_compensation_rad_per_m=self.torsion_compensation,shift_mapping='S=feed_turns*sqrt((2piR)^2+pitch^2)*chi, not LM*nt',high_level_feedback=False,initial_reference_max_error_m=self.initial_error_m,roll_sign=roll_sign)
        self.spec.update(moving_nominal_family=moving_family,anchor_frame=anchor_frame,bezier_bridge=bezier_bridge,local_material=local_material,smooth_source=smooth_source,discrete_transport=discrete_transport,nominal_spatial_phase='source=2pi*chi+angular_rate*S; target=2pi*chi-angular_rate*S; lower waypoint fixed' if moving_family else 'fixed world guide')
        self.spec.update(axial_family=axial_family)
        self.spec.update(release_curve=release_curve)
        if release_curve:self.spec['release_curve_rule']='first six material bend cells progressively released to0.4 then recovered by sigmoid(5*(nt-6)); nt=pi*0.058/0.0645*2.8*chi; 0.5cell smooth moving front. Scale actual Bishop curvature before integration; hold posterior end point/frame of unmodified template C, not actual pose.'
        if axial_family:self.spec['nominal_spatial_phase']='source=+angular_rate*S; target=-angular_rate*S; fixed-material circumferential angle cancels window advance, pipe-axis progress=a*S'

    def discrete_head_normal(self,points,locations,S,phases,angle):
        pp=list(points.copy());ll=list(locations.copy())
        # An imaginary posterior guide extension locates a source material
        # station after the physical body leaves source. It does not add rods.
        while .5*(ll[-1]+ll[-2])>self.start_arc-.15:
            ds=np.linspace(.1*LINK,2.5*LINK,70);trial=ll[-1]-ds;dist=np.linalg.norm(self.spline(trial)-pp[-1],axis=1);ids=np.flatnonzero(dist>=LINK)
            if not len(ids):raise ValueError('source gauge extension chord intersection failed')
            j=int(ids[0]);lo=trial[j];hi=ll[-1] if j==0 else trial[j-1]
            for _ in range(12):
                mid=(lo+hi)/2
                if np.linalg.norm(self.spline(mid)-pp[-1])>=LINK:lo=mid
                else:hi=mid
            ll.append((lo+hi)/2);pp.append(self.spline(ll[-1]))
            if len(pp)>120:raise ValueError('source material gauge extension exhausted')
        pp=np.array(pp);ll=np.array(ll);tangents=np.diff(pp,axis=0);tangents/=np.linalg.norm(tangents,axis=1)[:,None]
        station=.5*(ll[-2]+ll[-1]);path=np.linspace(self.anchor_arc,station,max(2,int(abs(station-self.anchor_arc)/.006)+1));tt=-self.spline(path,1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        normal=self.anchor_normal.copy();co,sn=np.cos(phases[0]),np.sin(phases[0]);normal[1:]=[co*normal[1]-sn*normal[2],sn*normal[1]+co*normal[2]];normal-=normal@tt[0]*tt[0];normal/=np.linalg.norm(normal);normal=self.transport_normal(normal,tt)
        normal=self.transport_normal(normal,np.vstack([tt[-1],tangents[-1]]))
        normal=np.cos(angle)*normal-np.sin(angle)*np.cross(tangents[-1],normal)
        return self.transport_normal(normal,tangents[::-1])

    @staticmethod
    def transport_normal(normal,tangents):
        for before,after in zip(tangents[:-1],tangents[1:]):
            cross=np.cross(before,after);sine=np.linalg.norm(cross);cosine=np.clip(before@after,-1,1)
            if sine>1e-12:
                axis=cross/sine;normal=normal*cosine+np.cross(axis,normal)*sine+axis*(axis@normal)*(1-cosine)
        normal-=normal@tangents[-1]*tangents[-1];return normal/np.linalg.norm(normal)

    def set_nominal_family(self,S):
        chi=S/self.feed_per_cycle;rate=1/np.sqrt(self.radius**2+(self.pitch/(2*np.pi))**2);phases=np.array([2*np.pi*chi+rate*S,2*np.pi*chi-rate*S])
        if self.axial_family:phases=np.array([rate*S,-rate*S])
        xyz=self.construction_xyz.copy();u=self.construction_u;cp=self.bridge_controls.copy()
        def rotate(points,axes,center,phase):
            points=points.copy();rad=points[:,axes]-center;co,sn=np.cos(phase),np.sin(phase);points[:,axes]=center+np.column_stack([co*rad[:,0]-sn*rad[:,1],sn*rad[:,0]+co*rad[:,1]]);return points
        source=u<=0;target=u>=self.bridge_parameter_end;xyz[source]=rotate(xyz[source],[1,2],np.array([0,1]),phases[0]);xyz[target]=rotate(xyz[target],[0,2],np.array([1.1,1]),phases[1])
        cp[:2]=rotate(cp[:2],[1,2],np.array([0,1]),phases[0]);cp[-2:]=rotate(cp[-2:],[0,2],np.array([1.1,1]),phases[1]);t0=self.bridge_start_t.copy();t1=self.bridge_end_t.copy()
        co,sn=np.cos(phases[0]),np.sin(phases[0]);t0[1:]=[co*t0[1]-sn*t0[2],sn*t0[1]+co*t0[2]];co,sn=np.cos(phases[1]),np.sin(phases[1]);t1[[0,2]]=[co*t1[0]-sn*t1[2],sn*t1[0]+co*t1[2]]
        mask=(u>0)&(u<self.bridge_parameter_end)
        if self.bezier_bridge:
            fraction=u[mask]/self.bridge_parameter_end;xyz[mask]=sum(math.comb(5,k)*(1-fraction[:,None])**(5-k)*fraction[:,None]**k*cp[k] for k in range(6))
        else:
            knots=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.bridge_controls,axis=0),axis=1))];bridge=CubicSpline(knots,cp,axis=0,bc_type=((1,t0),(1,t1)));xyz[mask]=bridge(u[mask])
        raw=CubicSpline(u,xyz,axis=0);grid=np.arange(u[0],u[-1],.001);p=raw(grid);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        self.spline=CubicSpline(arc,p,axis=0);self.start_arc=float(np.interp(0,grid,arc));self.target_arc=float(np.interp(self.bridge_parameter_end,grid,arc));return phases

    def geometric_frame(self,s):
        tangent=-self.spline(s,1);tangent/=np.linalg.norm(tangent)
        normal=self.spline(s,2);normal-=normal@tangent*tangent
        if np.linalg.norm(normal)<1e-8:raise ValueError('geometric material frame singular')
        normal/=np.linalg.norm(normal);return np.column_stack([tangent,normal,np.cross(tangent,normal)])

    def body(self,S,return_locations=False):
        head=self.start_arc+S;locations=[head];points=[self.spline(head)]
        for n in range(21):
            # First backward intersection, keeping material branch continuity.
            ds=np.linspace(.1*LINK,2.5*LINK,70);trial=locations[-1]-ds
            distances=np.linalg.norm(self.spline(trial)-points[-1],axis=1);ids=np.flatnonzero(distances>=LINK)
            if not len(ids):raise ValueError('no chord-length material intersection')
            j=int(ids[0]);lo=trial[j];hi=locations[-1] if j==0 else trial[j-1]
            for _ in range(12):
                mid=(lo+hi)/2
                if np.linalg.norm(self.spline(mid)-points[-1])>=LINK:lo=mid
                else:hi=mid
            locations.append((lo+hi)/2);points.append(self.spline(locations[-1]))
        return (np.array(points),np.array(locations)) if return_locations else np.array(points)

    def command(self,S):
        phases=self.set_nominal_family(S) if self.moving_family else np.zeros(2)
        chi=S/self.feed_per_cycle;points,locations=self.body(S,True);bone=SpatialBishopBackbone(points*1000,np.full(21,LINK*1000))
        angle=self.roll_sign*2*np.pi*chi-self.torsion_compensation*S;co,sn=np.cos(angle),np.sin(angle);rot=np.array([[co,sn],[-sn,co]])
        if not self.discrete_transport:
            tangent=-self.spline(self.start_arc+S,1);tangent/=np.linalg.norm(tangent)
            if self.anchor_frame:
                normal=self.anchor_normal.copy()
                if self.moving_family:
                    co,sn=np.cos(phases[0]),np.sin(phases[0]);normal[1:]=[co*normal[1]-sn*normal[2],sn*normal[1]+co*normal[2]]
                path=np.linspace(self.anchor_arc,self.start_arc+S,max(2,int((self.start_arc+S-self.anchor_arc)/.006)+1));tt=-self.spline(path,1);tt/=np.linalg.norm(tt,axis=1)[:,None]
                normal-=normal@tt[0]*tt[0];normal/=np.linalg.norm(normal);normal=self.transport_normal(normal,tt)
            elif self.moving_family:
                cross=np.cross(self.last_reference_tangent,tangent);sine=np.linalg.norm(cross);step=np.arctan2(sine,np.clip(self.last_reference_tangent@tangent,-1,1))
                if sine>1e-12:self.transported_normal=Rotation.from_rotvec(cross*step/sine).apply(self.transported_normal)
                self.transported_normal-=self.transported_normal@tangent*tangent;self.transported_normal/=np.linalg.norm(self.transported_normal);normal=self.transported_normal.copy();self.last_reference_tangent=tangent.copy()
            else:normal=self.normal_spline(self.start_arc+S)
            normal-=normal@tangent*tangent;normal/=np.linalg.norm(normal)
            normals=np.column_stack([normal,np.cross(tangent,normal)])@rot
        discrete_tangent=bone.start_frame[:,0]
        if self.discrete_transport:normal=self.discrete_head_normal(points,locations,S,phases,angle)
        else:normal=normals[:,0]-normals[:,0]@discrete_tangent*discrete_tangent;normal/=np.linalg.norm(normal)
        normals=np.column_stack([normal,np.cross(discrete_tangent,normal)])
        material_rotation=bone.start_frame[:,1:].T@normals
        field=bone.field@material_rotation;local_phase=np.zeros(len(field))
        if self.local_material:
            material_mid=.5*(bone.bounds[:-1]+bone.bounds[1:])/1000
            arc_mid=np.interp(material_mid,np.arange(22)*LINK,locations)
            weight=np.clip((arc_mid-self.start_arc)/(self.target_arc-self.start_arc),0,1);weight=weight*weight*(3-2*weight)
            local_phase=-weight*(phases[0]-phases[1]);co,sn=np.cos(local_phase),np.sin(local_phase);rotation=np.array([[co,sn],[-sn,co]]).transpose(2,0,1)
            field=np.einsum('ni,nij->nj',field,rotation)
        extra={}
        if self.release_curve:
            # This is an intentional curve deformation, not a gauge change.
            # Geometry still comes from G. No direct joint sinusoid is added.
            nt=np.pi*.058/LINK*2.8*chi;L=.4+.6/(1+np.exp(np.clip(-5*(nt-6),-60,60)))
            indices=np.arange(len(field))-1
            active=np.clip((nt-indices+.25)/.5,0,1);active=active*active*(3-2*active)
            weights=1-(1-L)*active*((indices>=0)&(indices<6))
            before=bone.sample(field=field,material_rotation=material_rotation,step_mm=8)
            field=field*weights[:,None]
            after=bone.sample(field=field,material_rotation=material_rotation,step_mm=8)
            rr=before['frames'][-1]@after['frames'][-1].T
            pp=np.asarray(after['points_mm']);pp=(pp-pp[-1])@rr.T+before['points_mm'][-1]
            cc=np.column_stack([np.interp(np.arange(22)*LINK*1000,after['s_mm'],pp[:,j]) for j in range(3)])/1000
            extra=dict(nominal_curve_points=cc,nominal_curve_head_frame=rr@after['frames'][0],release_weights=weights,release_nt=nt,release_L=L)
        q=bone.joint_angles(field=field)
        return q,dict(S=S,chi=chi,roll_phase=angle,nominal_pipe_phase=phases,local_material_phase=local_phase,curvature_field=field,material_rotation=material_rotation,reference_points=points,reference_locations=locations,reference_frame=np.column_stack([discrete_tangent,normals]),source_reference=np.mean(locations<self.start_arc),target_reference=np.mean(locations>self.target_arc),**extra)

    def global_points(self):return self.global_spline(np.linspace(self.global_start_arc-1.6,self.global_target_arc+2.0,900))
