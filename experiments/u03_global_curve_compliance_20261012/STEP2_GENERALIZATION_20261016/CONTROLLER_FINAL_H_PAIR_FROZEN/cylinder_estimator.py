"""Bounded contact cylinder candidates, with explicit identifiability gates."""
import numpy as np
from scipy.optimize import least_squares
from geometry import Geometry,normalize
from contact_sensor import known_source,receiver_axial_lower

class CylinderEstimator:
    def __init__(self,prior,source,settings):
        self.prior=prior.copy();self.geometry=prior.copy();self.source=source;self.settings=settings
        self.samples=[];self.next_fit=0.;self.first_contact=None;self.first_reliable=None
        self.probe_seen=np.zeros(2,bool);self.fit_history=[];self.last=dict(reliable=False,reason='no evidence',confidence=0.,sample_count=0)
        self.first_front_contact=None
    def observe(self,sensor,t,probe):
        new=[]
        for ct in sensor['contacts']:
            if known_source(ct,self.source):continue
            if ct['material']<8 and self.first_front_contact is None:self.first_front_contact=t
            axial,rad,rho=self.prior.coordinates(ct['point'])
            accepted=bool(receiver_axial_lower(self.prior,self.source)<axial<.90 and abs(rho-self.prior.radius)<.03 and abs(ct['normal']@self.prior.axis)<.45)
            item=dict(t=float(t),material=ct['material'],point=ct['point'].tolist(),normal=ct['normal'].tolist(),normal_N=ct['normal_N'],penetration_m=ct['penetration_m'],candidate_gate=accepted,source_excluded=False)
            new.append(item)
            if accepted:
                # Only transmitted, tracked excitation during measured receiver
                # candidate contact supplies probe evidence, never a request.
                self.probe_seen|=abs(probe)>.04
                self.samples.append(item)
                if self.first_contact is None:self.first_contact=t
        self.samples=[x for x in self.samples if t-x['t']<=self.settings['window_s']]
        if t>=self.next_fit:
            self.next_fit=t+self.settings['fit_interval_s']; self.fit(t)
        return new
    def fit(self,t):
        # Quantize only for evidence independence, not to change sensed geometry.
        unique={}
        for x in self.samples:
            key=tuple(np.round(np.asarray(x['point'])/.003).astype(int))+tuple(np.round(np.asarray(x['normal'])/.08).astype(int))
            unique[key]=x
        samples=list(unique.values());n=len(samples); mats=sorted(set(x['material'] for x in samples))
        result=dict(t=float(t),sample_count=n,material_ids=mats,probe_seen=self.probe_seen.tolist(),reliable=False,confidence=0.,reason='insufficient distinct evidence',axial_entry_observable=False,axial_entry_uncertainty_m=.04)
        if n<12 or len(mats)<2 or not self.probe_seen.all():
            self.last=result;self.fit_history.append(result);return
        points=np.array([x['point'] for x in samples]);normals=np.array([x['normal'] for x in samples]);normals/=np.linalg.norm(normals,axis=1)[:,None]
        values,vectors=np.linalg.eigh(normals.T@normals/n);axis=vectors[:,0]
        if axis@self.prior.axis<0:axis=-axis
        angle=float(np.degrees(np.arccos(np.clip(axis@self.prior.axis,-1,1))))
        axial_span=float(np.ptp(points@axis));result.update(normal_spectrum=values.tolist(),axial_span_m=axial_span,normal_axis_seed_angle_from_prior_deg=angle)
        if values[1]<.08 or axial_span<.035 or angle>self.settings['axis_bound_deg']:
            result['reason']='normal diversity, spatial span or bounded-axis gate';self.last=result;self.fit_history.append(result);return
        e1,e2=Geometry(self.prior.entry,axis,self.prior.radius).frame();basis=np.column_stack([e1,e2])
        uv=points@basis;nv=normals@basis
        A=np.zeros((2*n,3));A[::2,0]=1;A[1::2,1]=1;A[::2,2]=nv[:,0];A[1::2,2]=nv[:,1]
        fit0=np.linalg.lstsq(A,uv.ravel(),rcond=None)[0]
        center=fit0[:2]@basis.T+axis*(self.prior.entry@axis)
        prior_e1,prior_e2=self.prior.frame()
        def candidate(v):
            a=normalize(self.prior.axis+v[0]*prior_e1+v[1]*prior_e2)
            c=self.prior.entry+v[2]*prior_e1+v[3]*prior_e2
            # Maintain the prior axial entry plane, which side contacts cannot locate.
            c+=a*((self.prior.entry-c)@a)
            return Geometry(c,a,v[4])
        denom=max(axis@self.prior.axis,1e-6)
        v0=np.array([axis@prior_e1/denom,axis@prior_e2/denom,(center-self.prior.entry)@prior_e1,(center-self.prior.entry)@prior_e2,fit0[2]])
        ab=np.tan(np.radians(self.settings['axis_bound_deg']));cb=self.settings['center_bound_m'];rb=self.settings['radius_bound_m']
        lower=np.array([-ab,-ab,-cb,-cb,self.prior.radius-rb]);upper=np.array([ab,ab,cb,cb,self.prior.radius+rb]);v0=np.clip(v0,lower+1e-7,upper-1e-7)
        def residual(v):
            g=candidate(v);_,rad,rho=g.coordinates(points);rn=rad/np.maximum(rho[:,None],1e-9)
            return np.r_[rho-g.radius,.025*(normals@g.axis),.012*(rn-normals).ravel()]
        fit=least_squares(residual,v0,bounds=(lower,upper),loss='soft_l1',f_scale=.001,max_nfev=35)
        g=candidate(fit.x);_,rad,rho=g.coordinates(points);rn=rad/np.maximum(rho[:,None],1e-9)
        distance=abs(rho-g.radius); alignment=np.linalg.norm(rn-normals,axis=1)
        inliers=(distance<.0025)&(alignment<.18)
        rms=float(np.sqrt(np.mean(distance[inliers]**2))) if inliers.any() else float('inf')
        axis_rms=float(np.sqrt(np.mean((normals[inliers]@g.axis)**2))) if inliers.any() else float('inf')
        normal_rms=float(np.sqrt(np.mean(alignment[inliers]**2))) if inliers.any() else float('inf')
        angle=float(np.degrees(np.arccos(np.clip(g.axis@self.prior.axis,-1,1))))
        reliable=bool(inliers.sum()>=12 and inliers.mean()>.65 and rms<.0015 and axis_rms<.08 and normal_rms<.12 and angle<self.settings['axis_bound_deg'])
        # These are heuristic residual/conditioning diagnostics, not calibrated CIs.
        axis_unc=float(np.degrees(max(axis_rms,.003)/np.sqrt(max(n*values[1],1))))
        center_unc=float(max(rms,.0003)/np.sqrt(max(n*values[1],1)))
        result.update(reliable=reliable,reason='qualified cylinder' if reliable else 'inconsistent surface/cylinder fit',candidate=g.as_dict(),
            point_rms_m=rms,normal_axis_rms=axis_rms,normal_alignment_rms=normal_rms,inlier_count=int(inliers.sum()),
            inlier_fraction=float(inliers.mean()),axis_uncertainty_diagnostic_deg=axis_unc,transverse_center_uncertainty_diagnostic_m=center_unc,
            radius_uncertainty_diagnostic_m=max(rms,.0003)/np.sqrt(n),confidence=float(inliers.mean()*max(0.,1-rms/.0025)) if reliable else 0.,
            axis_angle_from_prior_deg=angle,latest_evidence_t=max(x['t'] for x in samples),independent_point_count=n)
        if reliable:
            self.geometry=g
            if self.first_reliable is None:self.first_reliable=t
        self.last=result;self.fit_history.append(result)

