"""Fixed rear nominal, fitted local material roll, finite connector compensation.

Nominal FK has its own data and never receives simulator state. Phase is not
material feed. The d=0 R2 reference is the sole initial geometry and prior.
"""
from pathlib import Path
import math
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from snake_control.local_material_reference import BoundaryMaterialCurve
from snake_control.executable_grasp_state import EquilibriumGraspController
from snake_control.three_segment_compliance import LocalSection


class FrontLocalRollingReference:
    maximum_phase_rad=math.radians(15.)
    def __init__(self,path,original_path):
        self.original=BoundaryMaterialCurve(original_path)
        self.initial=self.original.state(0.)
        with np.load(path) as z:
            self.phases=z['phase_rad'].copy();self.front_connector=z['joint_rad'].copy()
        self.interpolator=PchipInterpolator(self.phases,self.front_connector,axis=0)
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'))
        self.data=mujoco.MjData(self.model)
        self.qadr=self.model.jnt_qposadr[self.model.actuator_trnid[:,0]]
        self.anchor=self.model.body('link12').id
        self.set_private(self.initial)
        self.anchor_position=self.data.xpos[self.anchor].copy()
        self.anchor_orientation=self.data.xmat[self.anchor].reshape(3,3).copy()

    def __getattr__(self,name):return getattr(self.original,name)

    def set_private(self,x):
        self.data.qpos[:3]=x[:3]/1000
        quat=Rotation.from_rotvec(x[3:6]).as_quat();self.data.qpos[3:7]=quat[[3,0,1,2]]
        self.data.qpos[self.qadr]=x[6:]
        mujoco.mj_kinematics(self.model,self.data)

    def joint_angles(self,phase,roll_rad=0.):
        if roll_rad!=0 or not 0<=phase<=self.maximum_phase_rad:raise ValueError('Only registered local phase is allowed')
        q=self.initial[6:].copy()
        if phase==0.:return q
        # Consistent finite-chain fit. The continuum curvature phase is a guide,
        # while the fitted pipe-relative director twist defines nominal roll.
        q[:12]=self.interpolator(phase)
        return q

    def state(self,phase):
        if phase==0.:return self.initial.copy()
        x=self.initial.copy();x[6:]=self.joint_angles(phase)
        self.set_private(x)
        rotate=self.anchor_orientation@self.data.xmat[self.anchor].reshape(3,3).T
        x[:3]=(self.anchor_position+rotate@(x[:3]/1000-self.data.xpos[self.anchor]))*1000
        x[3:6]=(Rotation.from_matrix(rotate)*Rotation.from_rotvec(x[3:6])).as_rotvec()
        return x

    def nominal_root(self,phase):
        x=self.state(phase)
        return x[:3]-[1100.,0.,1000.],Rotation.from_rotvec(x[3:6]).as_matrix()

    def phase_derivative(self,phase):
        # Derivative of the jointly fitted finite front and connector.
        q=np.zeros(20)
        q[:12]=self.interpolator.derivative()(phase)
        return q

    def feed_tangent(self,phase):
        # Original holding projector retained by complete phase-zero branch.
        return self.original.feed_tangent(0.)


class FrontLocalRollingController(EquilibriumGraspController):
    def compute(self,measured,phase,elapsed,dt):
        if phase==0. and self.last_feed==0.:
            return super().compute(measured,0.,elapsed,dt)
        measured=np.asarray(measured,float)
        if measured.shape!=(20,) or not np.isfinite(measured).all():raise ValueError('20 finite encoders required')
        nominal=self.curve.joint_angles(phase)
        retained=self.last-self.curve.joint_angles(self.last_feed)
        p=self.forward_points(measured);pn=self.forward_points(nominal)
        _,orientation=self.curve.nominal_root(phase);orientation=self.registration@orientation
        centers=self.curve.lambda0-(np.arange(20)+1)*64.5
        smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
        weights=[smooth((-centers-64.5)/64.5),smooth((centers-self.curve.bounds[-1]-64.5)/64.5)]
        blend=smooth(elapsed/5.);phase_blend=smooth(phase/self.curve.maximum_phase_rad)
        old_tangent=self.curve.original.feed_tangent(0.)
        phase_tangent=self.curve.phase_derivative(phase)
        geometry=dict(config=self.config,points_mm=p,curve=dict(gamma_min_rad=math.radians(1.)))
        delta=np.zeros(20);update=np.zeros(20);axes=[]
        for side,(weight,world_axis) in enumerate(zip(weights,[np.array([1.,0,0]),np.array([0.,1.,0])])):
            ids=np.flatnonzero(weight>1e-8);lo,hi=int(ids[0]),int(ids[-1]+1)
            a,b=pn[lo:hi+2].copy(),p[lo:hi+2].copy();a-=a.mean(0);b-=b.mean(0)
            u,singular,v=np.linalg.svd(a.T@b)
            transport=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T
            rms=float(np.sqrt(np.mean(np.sum((a@transport.T-b)**2,axis=1))))
            if rms>64.5/4 or singular[1]/singular[0]<.05:raise ValueError('material cloud prior invalid')
            axis=transport@(orientation.T@world_axis)
            zero=LocalSection(geometry,self.zero,lo,hi,axis,np.zeros(3))
            shaped=LocalSection(geometry,self.parameters,lo,hi,axis,np.zeros(3))
            gauge=zero.align(measured[lo:hi],0.)
            difference=shaped.angles(gauge)-zero.angles(gauge);delta[lo:hi]=difference
            error=nominal[lo:hi]+retained[lo:hi]-measured[lo:hi]+self.restoring_error(retained[lo:hi],dt)
            change=blend*weight[lo:hi]*(difference-error)
            def projected(tangent):
                g=weight[lo:hi]*tangent[lo:hi]
                return change-g*float(g@change)/float(g@g) if float(g@g)>1e-12 else change
            old=projected(old_tangent)
            change=old if side==0 else (1-phase_blend)*old+phase_blend*projected(phase_tangent)
            update[lo:hi]=change
            axes.append(dict(joints=[lo+1,hi],direction_head=axis.tolist(),cloud_rms_mm=rms,
                side='source' if side==0 else 'target',phase_projector_blend=0. if side==0 else float(phase_blend)))
        retained[weights[0]+weights[1]<=1e-8]=0.
        raw=nominal+retained+update
        bounds=np.radians([[j['minimum_deg'],j['maximum_deg']] for j in self.config['joints']])
        signs=np.where(np.arange(20)%2,-1.,1.);low,high=np.sort(bounds*signs[:,None],axis=1).T
        limit=math.radians(10)*dt
        target=np.clip(raw,np.maximum(low,self.last-limit),np.minimum(high,self.last+limit))
        feasible=target-nominal;self.pending=target.copy(),float(phase)
        return target,dict(front_phase_rad=float(phase),feed_mm=0.,rear_phase_rad=0.,nominal_rad=nominal.tolist(),
            compliance_delta_rad=delta.tolist(),retained_shape_rad=feasible.tolist(),effective_residual_rad=feasible.tolist(),
            proposed_retained_rad=(retained+update).tolist(),shape_update_rad=(feasible-retained).tolist(),
            raw_target_rad=raw.tolist(),execution_projection_rad=(target-raw).tolist(),grasp_weights=(weights[0]+weights[1]).tolist(),
            material_axes=axes,phase_projector_blend=float(phase_blend),correction_errors=[],
            rate_limited_joints=int(np.count_nonzero(abs(raw-target)>1e-10)),
            angle_constrained_joints=int(np.count_nonzero((raw<low)|(raw>high))),restoring_time_constant_s=.5,
            feedback_fields=['encoder_angles','ACK_history','initial_geometry_prior'],
            limitation='rear nominal is fixed, actual rear anchoring is scored independently; world/contact never feeds control')
