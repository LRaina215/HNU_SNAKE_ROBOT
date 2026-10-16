"""Rough spatial route and original rolling curvature integration after capture."""
import numpy as np
from scipy.interpolate import CubicSpline
from native_mapping import NativeMapping,LINK
from geometry import normalize

class PostCaptureCurve(NativeMapping):
    def __init__(self,given,geometry,guide,head_frame,settings):
        self.given=given.copy();self.geometry=geometry.copy();self.capture_geometry=geometry.copy();self.capture_guide=guide.copy();self.original_capture_guide=guide.copy();self.capture_frame=head_frame.copy();self.current_geometry_theta=0.
        self.pitch=.18266863819422827;self.radius=.0991869522236036;self.feed_per_cycle=7.9099945262477895;self.feed_early=.054;self.feed_late=.09;self.material_roll_sign=settings['direct_extension']['post_material_roll_sign']
        self.receiver_winding_sign=settings['direct_extension']['receiver_winding_sign']
        self.nominal_frequency_hz=.02;self.supply_enabled=True;self.theta_offset=0.;self.delta=0.;self.sensor_geometry=geometry.copy()
        self.grow=1.;self.recover=1.;self.initial_normal=head_frame[:,1].copy();self.route_settings=settings
        from rough_planner import CurvatureCoordinates
        self.coordinates=CurvatureCoordinates()
        self.build_route()
    def set_geometry(self,geometry,relative_theta=0.):
        old=self.capture_geometry;e0,b0=old.frame();e,b=geometry.frame();ax,rad,_=old.coordinates(self.original_capture_guide);x=rad@e0;y=rad@b0
        co,sn=np.cos(relative_theta),np.sin(relative_theta);new=geometry.entry+ax[:,None]*geometry.axis+(co*x-sn*y)[:,None]*e+(sn*x+co*y)[:,None]*b
        boundary=self.route_settings['direct_extension']['wrap_hinges'];weight=np.clip(1-np.arange(22)/float(boundary),0,1)**2
        self.capture_guide=self.original_capture_guide+weight[:,None]*(new-self.original_capture_guide);self.geometry=geometry.copy();self.current_geometry_theta=relative_theta
    def build_route(self):
        g=self.geometry;e,b=g.frame();radius=g.radius+.039;head=self.capture_guide[0]
        axial,rad,_=g.coordinates(head);phi=np.arctan2(rad@b,rad@e);u=np.arange(.002,3.502,.002)
        beta=1/np.sqrt(radius**2+(self.pitch/(2*np.pi))**2);a=self.pitch*beta/(2*np.pi)
        angles=phi+self.receiver_winding_sign*beta*u;target=g.entry+(axial+a*u)[:,None]*g.axis+radius*(np.cos(angles)[:,None]*e+np.sin(angles)[:,None]*b)
        # Smooth 75mm lead preserves the prescribed capture curve at S=0.
        discrepancy=head-(g.entry+axial*g.axis+radius*(np.cos(phi)*e+np.sin(phi)*b));target+=(1-self.smooth(u/.075))[:,None]*discrepancy
        tail=self.given[-1];nextp=self.given[-2];theta=np.arctan2(tail[2]-1,tail[1]);r=np.linalg.norm(tail[1:]-[0,1])
        source_step=np.arctan2(nextp[2]-1,nextp[1])-theta;source_step=np.angle(np.exp(1j*source_step));dx=nextp[0]-tail[0]
        v=np.arange(-40,0);prefix=np.column_stack([tail[0]+dx*v,r*np.cos(theta+source_step*v),1+r*np.sin(theta+source_step*v)])
        xyz=np.vstack([prefix,self.capture_guide[::-1],target]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        self.spline=CubicSpline(arc,xyz,axis=0);self.capture_head_arc=float(arc[len(prefix)+21]);self.capture_tail_arc=float(arc[len(prefix)])
        self.start_arc=float(arc[len(prefix)+10]);self.target_arc=float(arc[len(prefix)+21]);self.nominal_target_arc=self.target_arc
        self.current_global_dense=xyz[::4];self.connector_arc=self.target_arc-self.start_arc
    def set_nominal_family(self,S):
        self.build_route();self.nominal_body_points,self.nominal_body_locations=self.body(S,True)
    def body(self,S,return_locations=False):
        # Head advances on a fixed spatial route. Every material chord stays64.5mm.
        loc=[self.capture_head_arc+self.feed_distance(S)];points=[self.spline(loc[0])]
        for _ in range(21):
            ds=np.linspace(.1*LINK,2.5*LINK,70);trial=loc[-1]-ds;dist=np.linalg.norm(self.spline(trial)-points[-1],axis=1);ids=np.flatnonzero(dist>=LINK)
            if not len(ids):raise ValueError('Post-capture route has no backward64.5mm material intersection')
            j=int(ids[0]);lo=trial[j];hi=loc[-1] if j==0 else trial[j-1]
            for _ in range(16):
                mid=(lo+hi)/2
                if np.linalg.norm(self.spline(mid)-points[-1])>=LINK:lo=mid
                else:hi=mid
            loc.append((lo+hi)/2);points.append(self.spline(loc[-1]))
        p=np.asarray(points);locations=np.asarray(loc);return (p,locations) if return_locations else p
    def material_anchor(self,S,locations):
        # At fixed spatial head station, parallel-transport the captured gauge then
        # advance the original native material rolling phase at0.02Hz.
        return self.capture_head_arc,self.initial_normal.copy(),self.material_roll_sign*2*np.pi*S/self.feed_per_cycle,0.
    def command(self,S):
        # Keep Step1's exact geometric curvature integration and q generation.
        # Represent those SAME native integrals on fixed sigma using the exact
        # integral-basis right inverse. Spatial/geometric station is separately
        # named; it is not substituted for hardware material length.
        q,legacy=NativeMapping.command(self,S);co=self.coordinates;field=co.field(q);info=dict(legacy)
        info.update(geometric_integration_arc_bounds_mm=legacy['curvature_bounds_mm'].copy(),
            geometric_integration_station_nodes_mm=legacy['material_arc_nodes_mm'].copy(),
            chi=self.material_roll_sign*S/self.feed_per_cycle,curvature_field=field,curvature_bounds_mm=co.bounds.copy(),material_arc_nodes_mm=co.material.copy(),
            spatial_guide_stations_m=legacy['reference_locations'].copy(),
            global_reference_points=self.nominal_body_points.copy(),global_reference_locations=self.nominal_body_locations.copy(),global_target_arc_m=self.nominal_target_arc,
            geometric_reference_points=legacy['reference_points'].copy(),geometric_curvature_field=field.copy(),geometric_native_command=q.copy(),
            material_metric='Fixed sigma=j*64.5mm exact native integral representation; Step1 q unchanged for the same spatial route and material gauge')
        return q,info
