"""Local bending in one Bishop frame, lengths mm and angles rad.

This is a new nominal shape family inspired by the local-window structure in
Rollinson/Choset. It is not the original 2023 controller or an EKF reproduction.
The same two bending components drive curve integration and Yamada joint angles.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss


class LocalBendBackbone:
    def __init__(self, helix, links, mu_mm=129., width_mm=64.5,
                 turn_rad=0., direction_rad=0., release=1.,reference_angles=None,reference_phase=0.):
        self.helix=helix
        self.links=np.asarray(links,float)
        self.bounds=np.r_[0.,np.cumsum(self.links)]
        self.mu=float(mu_mm)
        self.width=float(width_mm)
        self.turn=float(turn_rad)
        self.direction=float(direction_rad)
        self.release=float(release)
        if (self.links.ndim!=1 or len(self.links)<2 or not np.isfinite(self.links).all()
                or np.any(self.links<=0) or self.width<=0 or not all(math.isfinite(x) for x in
                (self.mu,self.width,self.turn,self.direction,self.release)) or not 0<=self.release<=1):
            raise ValueError('invalid local bending field')
        self.activation=min(abs(self.turn)/(math.pi/2),1.)*self.release
        self.normalizer=self.width*math.sqrt(math.pi/2)*(math.erf((self.bounds[-1]-self.mu)/(math.sqrt(2)*self.width))
                                            +math.erf(self.mu/(math.sqrt(2)*self.width)))
        if self.normalizer<=1e-10:
            raise ValueError('bending window does not overlap the body')
        self.reference_phase=float(reference_phase)
        self.correction=np.zeros((len(self.links),2))
        if reference_angles is not None:
            reference=np.asarray(reference_angles,float)
            if reference.shape!=(len(self.links)-1,) or not np.isfinite(reference).all():
                raise ValueError('invalid measured reference shape')
            difference=reference-helix.joint_angles(len(reference),self.links,self.reference_phase)
            for j,delta in enumerate(difference):
                component=1 if j%2==0 else 0
                self.correction[j:j+2,component]=-delta/(self.links[j]+self.links[j+1])

    def field(self,s,roll_phase):
        s=np.asarray(s,float)
        phase=self.helix.tau*s+roll_phase+self.helix.phase_offset
        window=np.exp(-.5*((s-self.mu)/self.width)**2)
        attenuation=1-self.activation*np.exp(-.5*((s-self.mu)/(2*self.width))**2)
        amount=self.turn*window/self.normalizer
        index=np.clip(np.searchsorted(self.bounds,s,side='right')-1,0,len(self.links)-1)
        correction=self.correction[index]
        rotation=roll_phase-self.reference_phase
        c,sine=math.cos(rotation),math.sin(rotation)
        b1=self.helix.kappa*np.cos(phase)+correction[...,0]*c-correction[...,1]*sine
        b2=self.helix.kappa*np.sin(phase)+correction[...,0]*sine+correction[...,1]*c
        return np.stack([b1*attenuation+amount*math.cos(self.direction),
                         b2*attenuation+amount*math.sin(self.direction)],axis=-1)

    def joint_angles(self,roll_phase):
        if not math.isfinite(roll_phase):
            raise ValueError('invalid rolling phase')
        if self.turn==0 and not self.correction.any():
            return self.helix.joint_angles(len(self.links)-1,self.links,roll_phase)
        nodes,weights=leggauss(32)
        left,right=self.bounds[:-1],self.bounds[1:]
        s=(left[:,None]+right[:,None])/2+(right-left)[:,None]/2*nodes
        components=self.field(s,roll_phase)
        per_link=np.sum(components*weights[None,:,None],axis=1)*(right-left)[:,None]/2
        integral=per_link[:-1]+per_link[1:]
        return -np.where(np.arange(len(integral))%2==0,integral[:,1],integral[:,0])

    def sample(self,roll_phase,step_mm=2.):
        if step_mm<=0:
            raise ValueError('invalid sampling interval')
        # State = position, tangent, the two Bishop normals. No division by
        # curvature: straight parts are supported without a Frenet singularity.
        initial=np.r_[np.zeros(3),[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]
        def derivative(s,state):
            b1,b2=self.field(s,roll_phase)
            tangent,e1,e2=state[3:6],state[6:9],state[9:12]
            return np.r_[tangent,b1*e1+b2*e2,-b1*tangent,-b2*tangent]
        s=np.linspace(0,self.bounds[-1],math.ceil(self.bounds[-1]/step_mm)+1)
        solution=solve_ivp(derivative,(0,self.bounds[-1]),initial,t_eval=s,
                           rtol=2e-9,atol=2e-10,max_step=min(2.,self.width/10))
        if not solution.success:
            raise ValueError(solution.message)
        return dict(s_mm=s,points_mm=solution.y[:3].T,tangent=solution.y[3:6].T,
                    normal1=solution.y[6:9].T,normal2=solution.y[9:12].T)
