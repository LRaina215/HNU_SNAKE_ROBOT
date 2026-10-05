"""Yield with retained support preload until measured transfer, then unload.

This experimental extension changes position references only. The preload
limit is the existing actuator's torque limit divided by position gain, not an
increase in torque capacity. Axes remain known simulation inputs in stage 1.
"""
import math
import numpy as np
from snake_control.segmented_pole_backbone import smoothstep
from snake_control.window_coordinated_winding import WindowCoordinatedWinding
from snake_control.spatial_bishop_backbone import SpatialBishopBackbone
from snake_control.three_segment_incremental import integrated_field


class SupportTransferWinding(WindowCoordinatedWinding):
    def __init__(self,*args,support_error_limit,**kwargs):
        super().__init__(*args,**kwargs)
        self.support_error_limit=np.asarray(support_error_limit)
        measured=np.asarray(kwargs['measured'])
        self.middle_preload=np.clip(self.last_target-measured,-self.support_error_limit,self.support_error_limit)
        self.previous_rear=self.rear

    def update_middle_reference(self,points,measured,dt):
        # Capture the load-bearing reference before a rear joint enters the
        # yielding middle. Previously ~96% of this error vanished in one step.
        if self.rear>self.previous_rear:
            ids=slice(self.previous_rear,self.rear)
            self.middle_preload[ids]=np.clip(self.last_target[ids]-measured[ids],
                -self.support_error_limit[ids],self.support_error_limit[ids])
        self.previous_rear=self.rear
        source_distance,target_distance,_,_=self.distances(points[1:-1])
        # Fade the support bias over one link as actual geometry changes pipe
        # association; assigning a joint to a new window alone does not erase it.
        retained=1-smoothstep(np.clip(.5+(source_distance-target_distance)/self.link,0,1))
        middle=self.weights[:,1]>0
        reference=measured+retained*self.middle_preload
        self.middle_reference[middle]+=-math.expm1(-dt/self.options.follow_seconds)*(reference[middle]-self.middle_reference[middle])
        return dict(support_reference_bias=self.middle_preload.copy(),support_retention=retained)

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            retained_preload='captured position error, bounded by existing actuator capacity',
            error_limit_rad=self.support_error_limit.tolist(),fade_distance_mm=self.link))


class AcquisitionRateWinding(WindowCoordinatedWinding):
    """Share one slower phase rate when formation gets ahead of acquisition.

    The previous coupling only made formation depend on rolling. Here actual
    acquisition also limits rolling, for BOTH ends, so the rear cannot keep
    supplying at full speed while the front is still forming. Shape feedback
    and A-D continue during the slower motion; no fixed rear pose is imposed.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.acquisition_phase=self.phase
        self.rolling_scale=1.

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        status=self.support(np.asarray(geometry['points_mm']),contact_window)
        lead=max(0.,self.progress-status['actual_prefix'])
        desired=float(np.clip(1-(lead/self.window.maximum_lead)**2,.1,1.))
        self.rolling_scale+=(1-math.exp(-dt/.4))*(desired-self.rolling_scale)
        demand=phase-self.acquisition_phase;self.acquisition_phase=phase
        virtual_phase=self.requested_phase+demand*self.rolling_scale
        q,info=super().compute(geometry,measured,virtual_phase,dt,contact_window,migrate)
        info.update(acquisition_lead=lead,shared_rolling_scale=self.rolling_scale)
        return q,info

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            coordination='shared rolling rate decreases as commanded formation leads measured acquisition',
            minimum_rate_fraction=.1,rate_filter_seconds=.4))


class AcquiredShapeWinding(WindowCoordinatedWinding):
    """Reduce nominal attraction after measured acquisition, retaining A-D.

    Incoming links need a forming reference. Keeping the same strong nominal
    recovery after acquisition can keep pulling against an already supported
    compliant shape. Fade to the existing 0.1 recovery after the ordered
    measured prefix passes each joint; this changes no actuator parameters.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.acquired_weight=np.zeros(self.n)
        self.acquired_weight[:self.initial_front]=1.

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        status=self.support(np.asarray(geometry['points_mm']),contact_window)
        acquired=np.arange(self.n)<status['actual_prefix']
        self.acquired_weight+=(1-math.exp(-dt/.6))*(acquired-self.acquired_weight)
        return super().compute(geometry,measured,phase,dt,contact_window,migrate)

    def rolling_observation(self,geometry,measured,observed):
        _,info=super().rolling_observation(geometry,measured,observed)
        gain=info['formation_gain']*(1-self.acquired_weight)+.1*self.formation_weights*self.acquired_weight
        gain[:self.initial_front]=.1*float(smoothstep(self.elapsed/1.6))
        change=gain[:,None]*(info['formation_nominal']-observed)
        info.update(formation_gain=gain,formation_change=change,acquired_weight=self.acquired_weight.copy())
        return observed+change,info

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            acquisition='ordered measured prefix, 0.6 s gain transition',
            acquired_recovery=.1,incoming_recovery=self.formation.recovery))


class BothEndsShapeWinding(AcquiredShapeWinding):
    """Maintain source wrapping while its remaining joints still roll there.

    Front formation already has nominal curvature recovery. Apply the same
    limited geometric term to the measured source helix; membership weights
    remove it as a joint enters the yielding middle. A-D and measured bending
    remain active at both ends, without changing actuator stiffness or limits.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        points=np.asarray(args[0]['points_mm'])[self.initial_rear:]-self.corner
        axis=-self.source
        radial=points-(points@axis)[:,None]*axis
        self.rear_winding=float(np.sign(np.sum(np.cross(radial[:-1],radial[1:])@axis)))
        self.rear_radius_reference=float(np.median(np.linalg.norm(radial,axis=1)))

    def rolling_observation(self,geometry,measured,observed):
        rolling,info=super().rolling_observation(geometry,measured,observed)
        points=np.asarray(geometry['points_mm'])
        bone=SpatialBishopBackbone(points,self.links)
        _,rotation=bone.material_field(measured)
        axis=-self.source
        relative=points[1:-1]-self.corner
        radial=relative-(relative@axis)[:,None]*axis
        distance=np.linalg.norm(radial,axis=1)
        ids=np.flatnonzero(self.weights[:,2]>.5)
        if len(ids):
            self.rear_radius_reference+=.15*(float(np.median(distance[ids]))-self.rear_radius_reference)
        h=abs(self.pitch_mm)/(2*np.pi)
        alpha=math.atan2(h,self.rear_radius_reference)
        kappa=self.rear_radius_reference/(self.rear_radius_reference**2+h*h)
        field=np.zeros_like(bone.field)
        for j,frame in enumerate(bone.mid_frames):
            normal=radial[j]/max(distance[j],1e-9)
            tangent=frame[:,0]
            desired=self.rear_winding*math.cos(alpha)*np.cross(axis,normal)+math.sin(alpha)*axis
            derivative=-kappa*normal+(desired-tangent*(desired@tangent))/(2*self.link)
            field[j+1]=derivative@(frame[:,1:]@rotation)
        nominal=integrated_field(bone,field)
        gain=.1*float(smoothstep(self.elapsed/1.6))*self.weights[:,2]
        change=gain[:,None]*(nominal-observed)
        info.update(rear_formation_change=change,rear_formation_radius_mm=self.rear_radius_reference,
                    rear_formation_gain=gain)
        return rolling+change,info

    def configuration(self):
        config=super().configuration()
        config['support_transfer'].update(rear_recovery=.1,rear_winding=self.rear_winding,
            rear_shape='measured source-axis helix, removed on entering the yielding middle')
        return config


class ContactReferenceWinding(AcquiredShapeWinding):
    """Project position references to avoid opening the last source contacts.

    A kinematic first-order constraint is evaluated with the current floating
    base pose. It changes joint targets only: no simulated weld, force or pose
    correction is applied. Contact truth is a stage-1 input, and this local
    reference constraint is not a guarantee of actual dynamic contact.
    """
    def __init__(self,*args,local_contact_projection=False,dual_contact_projection=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.local_contact_projection=local_contact_projection
        self.dual_contact_projection=dual_contact_projection

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        from scipy.optimize import minimize
        previous=self.last_target.copy()
        q,info=super().compute(geometry,measured,phase,dt,contact_window,migrate)
        ids=np.flatnonzero(np.asarray(contact_window)[-1,0]>.1)
        if self.local_contact_projection and not self.dual_contact_projection:
            # Only downstream contacts can be changed while front targets are
            # held. An already acquired front link may still touch the source
            # junction; treating that contact as a rear anchor makes the local
            # constraint infeasible regardless of any downstream joint motion.
            ids=ids[ids>self.front]
        protect=len(ids)>0 and info['actual_prefix']<12 and not self.handed_off
        if self.local_contact_projection or self.dual_contact_projection:
            protect=protect and len(ids)<=2
        if not protect:
            info.update(source_reference_projected=False)
            return q,info
        target_ids=np.flatnonzero(np.asarray(contact_window)[-1,1]>.1) if self.dual_contact_projection else np.array([],int)
        def radii(angles):
            points=self.forward_points(angles)
            centers=(points[:-1]+points[1:])/2
            return np.r_[np.linalg.norm(np.cross(centers[ids]-self.corner,self.source),axis=1),
                np.linalg.norm(np.cross(centers[target_ids]-self.corner,self.target),axis=1)]
        base=radii(measured)
        jac=np.zeros((len(base),self.n));eps=1e-5
        for j in range(self.n):
            perturbed=measured.copy();perturbed[j]+=eps
            jac[:,j]=(radii(perturbed)-base)/eps
        # Existing contact can roll/slide tangentially; its linearised normal
        # displacement may open by at most 0.5 mm in the next reference.
        budget=math.radians(self.window.target_lead_deg)
        low=np.maximum(measured-budget,previous-self.options.maximum_speed_rad_s*dt)
        high=np.minimum(measured+budget,previous+self.options.maximum_speed_rad_s*dt)
        signs=np.where(np.arange(self.n)%2,-1.,1.)
        physical=np.sort(np.radians([[j['minimum_deg'],j['maximum_deg']] for j in geometry['config']['joints']])*signs[:,None],axis=1)
        low=np.maximum(low,physical[:,0]+.01);high=np.minimum(high,physical[:,1]-.01)
        if self.local_contact_projection and not self.dual_contact_projection:
            # Leave the established front's A-D/rolling targets unchanged.
            # Otherwise a head-anchored FK constraint can alter that front to
            # satisfy a distant source contact, while the physical head floats.
            low[:self.front]=q[:self.front];high[:self.front]=q[:self.front]
        # If initial tracking error already makes the two bounds inconsistent,
        # retain the existing conditioned command rather than fake feasibility.
        if np.any(low>high):
            info.update(source_reference_projected=False,projection_reason='incompatible slew and tracking bounds')
            return q,info
        result=minimize(lambda x:.5*float((x-q)@(x-q)),np.clip(q,low,high),
            jac=lambda x:x-q,bounds=list(zip(low,high)),
            constraints=[dict(type='ineq',fun=lambda x:.5-jac@(x-measured),jac=lambda x:-jac)],
            method='SLSQP',options=dict(ftol=1e-10,maxiter=60))
        if result.success:
            self.last_target=result.x.copy()
            info.update(source_reference_projected=True,source_reference_module_ids=ids,
                target_reference_module_ids=target_ids,projection_delta=result.x-q,
                projected_source_opening_mm=(jac@(result.x-measured))[:len(ids)],
                projected_target_opening_mm=(jac@(result.x-measured))[len(ids):])
            return result.x,info
        info.update(source_reference_projected=False,projection_reason=result.message)
        return q,info

    def configuration(self):
        config=super().configuration()
        config['support_transfer'].update(source_reference='linearised module radial constraint until measured prefix 12',
            source_opening_allowance_mm=.5,extra_input='supplied source-support observations and FK; runner records whether inferred or simulation truth',
            local_contact_projection=self.local_contact_projection,dual_contact_projection=self.dual_contact_projection)
        return config


class CapacityPreloadWinding(ContactReferenceWinding):
    """Capture attainable preload instead of rotating a saturated angle error.

    Clipping the initial position error to torque_limit / position_gain gives
    the SAME initial actuator torque for this model. Reconstruct its material
    field and retain that finite preload. This does not cap forming/rolling
    motion to that tiny angle and does not modify the saved physical state.
    """
    def __init__(self,*args,support_error_limit,**kwargs):
        super().__init__(*args,**kwargs)
        measured=np.asarray(kwargs['measured'])
        self.error_limit=np.asarray(support_error_limit)
        effective=measured+np.clip(np.asarray(args[2])-measured,-self.error_limit,self.error_limit)
        observed=self.material_integrals(args[0]['points_mm'],measured)
        self.preload=self.material_integrals(self.forward_points(effective),effective)-observed

    def configuration(self):
        config=super().configuration()
        config['support_transfer'].update(preload='initial torque-equivalent angle residual in material coordinates',
            preload_error_limit_rad=self.error_limit.tolist())
        return config


class RadialAcquisitionWinding(CapacityPreloadWinding):
    """Close incoming radial gaps using the already contacting front as guide.

    A correct circumferential tangent alone does not ensure a new link reaches
    the same cylinder as the supported front. Add a measured radial-error
    curvature term over one link length. Its reference radius is measured at
    current target contacts; the original A-D correction remains active.
    """
    def __init__(self,*args,coordinate_acquisition=False,acquire_initial_front=False,bootstrap_front=False,coherent_acquisition=False,loosen_unacquired_front=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.coordinate_acquisition=coordinate_acquisition
        self.acquire_initial_front=acquire_initial_front
        self.bootstrap_front=bootstrap_front
        self.coherent_acquisition=coherent_acquisition
        self.loosen_unacquired_front=loosen_unacquired_front
        self.initial_front_acquired=np.zeros(self.initial_front)
        self.acquisition_phase=self.phase
        self.rolling_scale=1.

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        points=np.asarray(geometry['points_mm'])
        centers=(points[:-1]+points[1:])/2
        radial=np.linalg.norm(np.cross(centers-self.corner,self.target),axis=1)
        contacts=np.asarray(contact_window)[-1,1]>.1
        axial=(centers-self.corner)@self.target
        usable=contacts&(axial>self.radius+self.options.shell_radius_mm)
        if np.any(usable):
            value=float(np.median(radial[usable]))
            if not hasattr(self,'contact_radius'):self.contact_radius=value
            self.contact_radius+=(1-math.exp(-dt/.4))*(value-self.contact_radius)
            vertices=np.linalg.norm(np.cross(points-self.corner,self.target),axis=1)
            ids=np.flatnonzero(usable)
            # Joint vertices sit outside their contacting link-centre radius.
            # Use the same contacting links to obtain a polygon-compatible
            # forming radius; a loose free head must not inflate this guide.
            joint_value=float(np.median(np.r_[vertices[ids],vertices[ids+1]]))
            if not hasattr(self,'contact_joint_radius'):self.contact_joint_radius=joint_value
            self.contact_joint_radius+=(1-math.exp(-dt/.4))*(joint_value-self.contact_joint_radius)
        self.module_radii=radial
        if self.bootstrap_front and hasattr(self,'contact_radius'):
            acquired=(radial[:self.initial_front]<self.contact_radius+15.)&(axial[:self.initial_front]>self.radius/2)
            self.initial_front_acquired+=(1-math.exp(-dt/.6))*(acquired-self.initial_front_acquired)
        coordination={}
        if self.coordinate_acquisition:
            # Slow BOTH phase clocks when measured recruitment falls behind
            # the forming window. Shape/A-D feedback keeps updating, while
            # the rear no longer rolls out at full rate during that delay.
            status=self.support(points,contact_window)
            lead=max(0.,self.progress-status['actual_prefix'])
            desired=float(np.clip(1-(lead/self.window.maximum_lead)**2,.1,1.))
            self.rolling_scale+=(1-math.exp(-dt/.4))*(desired-self.rolling_scale)
            demand=phase-self.acquisition_phase;self.acquisition_phase=phase
            phase=self.requested_phase+demand*self.rolling_scale
            coordination=dict(acquisition_lead=lead,shared_rolling_scale=self.rolling_scale)
        q,info=super().compute(geometry,measured,phase,dt,contact_window,migrate)
        return q,dict(**info,**coordination)

    def formation_radius(self,distance):
        if self.bootstrap_front and hasattr(self,'contact_joint_radius'):
            return self.contact_joint_radius
        return super().formation_radius(distance)

    def rolling_observation(self,geometry,measured,observed):
        rolling,info=super().rolling_observation(geometry,measured,observed)
        if self.bootstrap_front:
            gain=info['formation_gain'].copy()
            gain[:self.initial_front]=float(smoothstep(self.elapsed/1.6))*(
                self.formation.recovery*(1-self.initial_front_acquired)+.1*self.initial_front_acquired)
            change=gain[:,None]*(info['formation_nominal']-observed)
            rolling=observed+change
            info.update(formation_gain=gain,formation_change=change,
                initial_front_acquired=self.initial_front_acquired.copy())
        if self.coherent_acquisition:
            # The local radial/tangent field can fold the free head back
            # toward the junction while its cross-section still looks wrapped.
            # Test one Yamada curvature/torsion pattern across the currently
            # forming front, using its measured phase and contact-based radius.
            a,b=self.helix_basis(self.radius_reference,self.pitch_sign)
            weight=np.sqrt(self.formation_weights)
            fit=np.linalg.lstsq(np.column_stack((a,b))*weight[:,None],measured*weight,rcond=None)[0]
            theta=math.atan2(fit[1],fit[0])
            q=a*math.cos(theta)+b*math.sin(theta)
            quadrature=-a*math.sin(theta)+b*math.cos(theta)
            odd=np.arange(self.n)%2
            nominal=np.column_stack((np.where(odd,-q,-quadrature),np.where(odd,quadrature,-q)))
            change=info['formation_gain'][:,None]*(nominal-observed)
            rolling=observed+change
            info.update(formation_nominal=nominal,formation_change=change,
                formation_helix_phase=theta,formation_helix_angles=q)
        loosen=np.ones(self.n)
        if self.loosen_unacquired_front:
            # CMU's initial bend window uses reduced amplitude before it wraps
            # the target. Test that principle with our measured acquisition,
            # not its time/window formula: only the initially draped free head
            # is eased. Supported links recover the original forming field.
            # This is a curvature-reference change, not a stiffness change;
            # A-D feedback, measured-field rolling and delivery limits remain.
            loosen[:self.initial_front]=.4+.6*self.initial_front_acquired
            nominal=info['formation_nominal']*loosen[:,None]
            change=info['formation_gain'][:,None]*(nominal-observed)
            rolling=observed+change
            info.update(formation_nominal=nominal,formation_change=change,
                        initial_front_loosen_factor=loosen)
        if not hasattr(self,'contact_radius'):return rolling,info
        points=np.asarray(geometry['points_mm'])
        bone=SpatialBishopBackbone(points,self.links)
        _,rotation=bone.material_field(measured)
        rel=points[1:-1]-self.corner
        radial=rel-(rel@self.target)[:,None]*self.target
        normal=radial/np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
        gap=np.maximum(0.,(self.module_radii[:-1]+self.module_radii[1:])/2-self.contact_radius)
        field=np.zeros_like(bone.field)
        for j,frame in enumerate(bone.mid_frames):
            derivative=-gap[j]/self.link**2*normal[j]
            field[j+1]=derivative@(frame[:,1:]@rotation)
        gain=self.formation.recovery*self.formation_weights
        # A natural drape may have only a single target contact. Membership in
        # the initial front is not evidence that those links are already close
        # to the pipe; apply the same radial feedback during their acquisition.
        gain[:self.initial_front]=self.formation.recovery*float(smoothstep(self.elapsed/1.6)) if self.acquire_initial_front or self.bootstrap_front else 0.
        change=(gain*loosen)[:,None]*integrated_field(bone,field)
        info.update(contact_radius_mm=self.contact_radius,incoming_radial_gap_mm=gap,
                    radial_acquisition_change=change)
        return rolling+change,info

    def configuration(self):
        config=super().configuration()
        config['support_transfer'].update(acquisition_curvature='positive radial gap / link_length^2',
            radial_reference='median radius of supplied target-support modules beyond the junction',
            coordinate_acquisition=self.coordinate_acquisition,acquire_initial_front=self.acquire_initial_front,
            bootstrap_front=self.bootstrap_front,coherent_acquisition=self.coherent_acquisition,
            loosen_unacquired_front=self.loosen_unacquired_front)
        return config


class RateMatchedMiddleWinding(WindowCoordinatedWinding):
    """Let the connector yield at the current helix's nominal rolling rate.

    An unconstrained angle follower tracks a falling connector almost as fast
    as it falls. Bound reference motion, while the existing position actuator
    can resist the excess. No physical damping or torque limit is changed.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        middle=self.weights[:,1]>0
        self.middle_reference[middle]=np.asarray(kwargs['measured'])[middle]
        self.middle_phase=self.phase

    def update_middle_reference(self,points,measured,dt):
        phase_step=abs(self.phase-self.middle_phase);self.middle_phase=self.phase
        a,b=self.helix_basis(self.radius_reference,self.pitch_sign)
        budget=np.hypot(a,b)*phase_step
        middle=self.weights[:,1]>0
        request=-math.expm1(-dt/self.options.follow_seconds)*(measured-self.middle_reference)
        change=np.clip(request,-budget,budget)
        self.middle_reference[middle]+=change[middle]
        return dict(middle_reference_rate_rad_s=budget/dt,
            middle_rate_limited=int(np.count_nonzero(middle&(abs(request)>budget))))

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            middle='measured angle follower with helix-amplitude times rolling-rate slew bound'))


class CoherentHelixWinding(WindowCoordinatedWinding):
    """Recruit into one helix pattern fitted to the already draped front.

    Reuse the analytic Yamada helix basis for both quadrature components. The
    previous spatial derivative recovery was evaluated independently along
    each incoming link; here all admitted links share curvature, torsion and
    the measured front's phase gauge. Measured bending and A-D remain active.
    """
    def formation_phase(self,a,b,measured):
        fit=np.linalg.lstsq(np.column_stack([a,b])[:self.initial_front],measured[:self.initial_front],rcond=None)[0]
        return math.atan2(fit[1],fit[0])

    def rolling_observation(self,geometry,measured,observed):
        points=np.asarray(geometry['points_mm'])[1:-1]-self.corner
        radius=float(np.median(np.linalg.norm(np.cross(points[:self.initial_front],self.target),axis=1)))
        self.radius_reference+=.15*(radius-self.radius_reference)
        a,b=self.helix_basis(self.radius_reference,self.pitch_sign)
        theta=self.formation_phase(a,b,measured)
        q=a*math.cos(theta)+b*math.sin(theta)
        quadrature=a*math.cos(theta+math.pi/2)+b*math.sin(theta+math.pi/2)
        odd=np.arange(self.n)%2
        nominal=np.column_stack((np.where(odd,-q,-quadrature),np.where(odd,quadrature,-q)))
        gain=self.formation.recovery*self.formation_weights
        gain[:self.initial_front]=self.window.supported_recovery*float(smoothstep(self.elapsed/1.6))
        change=gain[:,None]*(nominal-observed)
        return observed+change,dict(formation_weights=self.formation_weights.copy(),
            formation_radius_mm=self.radius_reference,formation_pitch_sign=self.pitch_sign,
            formation_nominal=nominal,formation_observed=observed,formation_change=change,
            formation_gain=gain,formation_helix_phase=theta,formation_helix_angles=q)

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            formation='one analytic Yamada helix pattern, measured-front phase alignment'))


class AnchoredHelixWinding(CoherentHelixWinding):
    """Keep the front helix's phase state while admitting successive joints.

    Fitting a fresh phase to every measured frame changes the waveform that
    incoming joints must join. Here alignment happens once at takeover, then
    the same cumulative rolling phase advances the existing and forming front.
    Radius feedback and the measured A-D correction remain active.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        a,b=self.helix_basis(self.radius_reference,self.pitch_sign)
        self.anchor_phase=super().formation_phase(a,b,np.asarray(kwargs['measured']))
        self.observed_front_phase=self.phase

    def formation_phase(self,a,b,measured):
        # The parent applies this step's rolling rotation AFTER this hook.
        phase=self.anchor_phase+self.observed_front_phase-self.initial_phase
        self.observed_front_phase=self.phase
        return phase

    def configuration(self):
        config=super().configuration()
        config['support_transfer'].update(phase='initial alignment plus accumulated front roll; no per-frame refit',
            anchor_phase=self.anchor_phase)
        return config


class TurningFormationWinding(WindowCoordinatedWinding):
    """Form along a turning local axis before joining the target helix.

    A recruited joint can still physically surround the source pipe. Directly
    imposing target-axis curvature there can remove its source support. Blend
    the nominal local axis across four links beyond the measured ordered cut,
    including the axis-rotation contribution to tangent curvature. Existing
    material rolling, measured A-D and yielding-middle rules are retained.
    """
    def guide_boundary(self,centers):
        ds,dr,_,_=self.distances(centers)
        cuts=np.arange(self.initial_front,self.n+1)
        return int(cuts[np.argmin([dr[:k].sum()+ds[k:].sum() for k in cuts])])

    def rolling_observation(self,geometry,measured,observed):
        points=np.asarray(geometry['points_mm'])
        centers=(points[:-1]+points[1:])/2
        cut=self.guide_boundary(centers)
        # This cut chooses reference geometry only. The existing independent
        # full-transfer audit still requires all 21 modules and support.
        width=4.
        x=np.clip((np.arange(self.n)-(cut-1))/width,0.,1.)
        mix=smoothstep(x)
        axes=-(1-mix)[:,None]*self.target-mix[:,None]*self.source
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        spin=np.cross(axes,np.gradient(axes,self.link,axis=0))
        rel=points[1:-1]-self.corner
        cp_target=(rel@self.target)[:,None]*self.target
        cp_source=(rel@self.source)[:,None]*self.source
        radial=rel-((1-mix)[:,None]*cp_target+mix[:,None]*cp_source)
        radial-=np.sum(radial*axes,axis=1)[:,None]*axes
        distance=np.linalg.norm(radial,axis=1)
        self.radius_reference+=.15*(float(np.median(distance[:self.initial_front]))-self.radius_reference)
        h=abs(self.pitch_mm)/(2*np.pi)
        alpha=math.atan2(h,self.radius_reference)
        kappa=self.radius_reference/(self.radius_reference**2+h*h)
        bone=SpatialBishopBackbone(points,self.links)
        _,rotation=bone.material_field(measured)
        field=np.zeros_like(bone.field)
        for j,frame in enumerate(bone.mid_frames):
            normal=radial[j]/max(distance[j],1e-9)
            desired=self.winding*math.cos(alpha)*np.cross(axes[j],normal)+math.sin(alpha)*axes[j]
            tangent=frame[:,0]
            derivative=np.cross(spin[j],desired)-kappa*normal+(desired-tangent*(desired@tangent))/(2*self.link)
            field[j+1]=derivative@(frame[:,1:]@rotation)
        nominal=integrated_field(bone,field)
        gain=self.formation.recovery*self.formation_weights
        gain[:self.initial_front]=self.window.supported_recovery*float(smoothstep(self.elapsed/1.6))
        change=gain[:,None]*(nominal-observed)
        return observed+change,dict(formation_weights=self.formation_weights.copy(),
            formation_radius_mm=self.radius_reference,formation_pitch_sign=self.pitch_sign,
            formation_nominal=nominal,formation_observed=observed,formation_change=change,
            formation_gain=gain,turning_axis_cut=cut,turning_axis_source_weight=mix,
            turning_axis_spin=spin)

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            formation='turning nominal axis over four links, including axis-rotation tangent derivative',
            geometry_cut='ordered minimum of module-axis distances, reference only'))


class SmoothTurningFormationWinding(TurningFormationWinding):
    """Prevent a geometry-label jump from instantly moving the reference bend."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guide_cut=float(self.initial_front)
        self.guide_candidate=None;self.guide_streak=0

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        self.guide_dt=dt
        self.guide_contacts=np.asarray(contact_window)[-1,1]>.1
        return super().compute(geometry,measured,phase,dt,contact_window,migrate)

    def guide_boundary(self,centers):
        candidate=super().guide_boundary(centers)
        corroborated=bool(np.any(self.guide_contacts[max(0,candidate-3):candidate+1]))
        if self.handed_off:candidate=self.n;corroborated=True
        self.guide_streak=self.guide_streak+1 if candidate==self.guide_candidate and corroborated else int(corroborated)
        self.guide_candidate=candidate
        if self.guide_streak>=3:
            self.guide_cut+=float(np.clip(candidate-self.guide_cut,-1.25*self.guide_dt,1.25*self.guide_dt))
        return self.guide_cut

    def configuration(self):
        config=super().configuration()
        config['support_transfer'].update(geometry_cut='contact-corroborated ordered cut, three frames, maximum 1.25 joints/s')
        return config


class ReferenceMemoryWinding(WindowCoordinatedWinding):
    """Advance a compliant bending reference instead of restoring old preload.

    B_next before rolling = B_ref + a(B_measured-B_ref)
                            + g(B_helix-B_ref).
    The existing measured A-D difference is then added by the parent. The
    reference comes from the last issued position target, so yielding into a
    new window does not recover a stale captured torque direction. This is a
    control-state update; robot stiffness and torque capacity are unchanged.
    """
    def rolling_observation(self,geometry,measured,observed):
        _,info=super().rolling_observation(geometry,measured,observed)
        reference=self.material_integrals(self.forward_points(self.last_target),self.last_target)
        # 0.15 at the established 0.2 s update; use a time constant when dt
        # changes. The 2023 A-D correction still uses current measured shape.
        dt=self.elapsed-getattr(self,'reference_elapsed',0.)
        self.reference_elapsed=self.elapsed
        alpha=1-math.exp(math.log(.85)*dt/.2)
        gain=info['formation_gain']
        rolling=reference+alpha*(observed-reference)+gain[:,None]*(info['formation_nominal']-reference)
        self.preload[:]=0.
        info.update(formation_change=rolling-observed,reference_field=reference,
                    reference_measurement_gain=alpha)
        return rolling,info

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            reference='issued bending reference, measured-shape relaxation and A-D feedback',
            reference_measurement_gain_at_point2_seconds=.15))


class SupportHoldWinding(WindowCoordinatedWinding):
    """Diagnostic: preserve the last rear pose during weak support handover.

    A simulation-only static lower bound vetoes rear release. This tests
    actual position-reference holding, which a pause in rolling phase alone
    cannot provide in a measured-shape controller. It is not an autonomous
    support estimator and does not change the robot's actuator model.
    """
    def __init__(self,*args,support_bound,hold_bound=1.,bound_held_phase=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.support_bound=support_bound
        self.hold_bound=hold_bound
        self.bound_held_phase=bound_held_phase
        self.external_phase=self.phase
        self.holding=False;self.hold_weight=0.
        self.hold_reference=self.last_target.copy()

    def rear_release_permitted(self):
        return not self.holding

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        bound=self.support_bound()
        ratio=bound['lower_bound']
        source_count=int(np.count_nonzero(np.asarray(contact_window)[-1,0]>.1))
        insufficient=ratio is None or ratio>self.hold_bound
        if not self.holding and 0<source_count<=2 and insufficient:
            self.holding=True;self.hold_reference=self.last_target.copy()
        elif self.holding and ratio is not None and ratio<.8*self.hold_bound:
            self.holding=False
            # Re-enter from the issued pose; do not restore the old residual.
            self.active[self.weights[:,2]>0]=False
        self.hold_weight+=np.clip(float(self.holding)-self.hold_weight,-dt/.4,dt/.4)
        self.rear_motion_scale=1-self.hold_weight
        previous=self.last_target.copy()
        demand=phase-self.external_phase;self.external_phase=phase
        if self.bound_held_phase:
            lag=max(0.,self.rear_phase-self.phase)
            room=float(np.clip(1-lag/(math.pi/4),0,1))
            phase=self.requested_phase+demand*(1-self.hold_weight*(1-room))
        q,info=super().compute(geometry,measured,phase,dt,contact_window,migrate)
        weight=self.weights[:,2]*self.hold_weight
        q=(1-weight)*q+weight*self.hold_reference
        q=previous+np.clip(q-previous,-self.options.maximum_speed_rad_s*dt,self.options.maximum_speed_rad_s*dt)
        self.last_target=q.copy()
        info.update(rear_reference_held=self.holding,rear_hold_weight=self.hold_weight,
            target_only_static_bound=ratio,source_contact_count=source_count)
        return q,info

    def configuration(self):
        return dict(**super().configuration(),support_transfer=dict(
            rear_hold='issued rear pose when <=2 source modules and target-only optimistic static bound exceeds threshold',
            hold_bound=self.hold_bound,resume_bound=.8*self.hold_bound,
            bounded_phase_while_held=self.bound_held_phase,
            extra_privileged_input='MuJoCo contact Jacobians, mass and friction bounds; diagnostic only'))
