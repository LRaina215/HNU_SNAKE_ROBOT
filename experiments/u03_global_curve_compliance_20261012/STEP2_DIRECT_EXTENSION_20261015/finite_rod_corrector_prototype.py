"""Local finite-rod correction around a native curve, using prior geometry.

This private kinematic model contains only known source/robot and a synthetic
prior receiver. It is never integrated. No physical root or true target query.
"""
import numpy as np
from scipy.optimize import least_squares
from rough_planner import coarse_model
from native_robot import NativeRobot
from physics import mujoco

class FiniteRodCorrector:
    def __init__(self,geometry,data):
        self.last=None;self.geometry=None;self.set_geometry(geometry,data)
    def set_geometry(self,geometry,data):
        signature=tuple(np.r_[geometry.entry,geometry.axis,geometry.radius])
        if signature==self.geometry:return
        self.m=coarse_model(geometry);self.robot=NativeRobot(self.m,data);self.geometry=signature
        self.target=self.m.geom('rough_receiver').id;self.source=[self.m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']]
        self.geoms=[int(self.m.body_geomadr[self.robot.head+j]) for j in range(21)]
    def solve(self,data,anchor,native,points,nfree):
        robot=self.robot;robot.observe(data,anchor);base=data.qpos[robot.qa].copy();base[:nfree]=native[:nfree];initial=base[:nfree].copy();bound=np.radians(88);start=np.clip(initial,-bound,bound)
        def residual(x):
            base[:nfree]=x;pred=robot.fk(base);position=((pred[:nfree]-points[:nfree])/.010).ravel();gaps=[]
            for j in range(nfree):
                to=np.zeros(6);gap=mujoco.mj_geomDistance(self.m,robot.d,self.geoms[j],self.target,.05,to);gaps.append(min(0.,gap+.0008)/.002)
                for source in self.source:
                    gap=mujoco.mj_geomDistance(self.m,robot.d,self.geoms[j],source,.05,to);gaps.append(min(0.,gap+.0015)/.002)
            return np.r_[position,.2*(x-initial)/np.radians(15),gaps]
        result=least_squares(residual,start,bounds=(-bound*np.ones(nfree),bound*np.ones(nfree)),max_nfev=18,ftol=1e-4,xtol=1e-4,gtol=1e-4)
        base[:nfree]=result.x;actual=robot.fk(base);self.last=dict(nfev=int(result.nfev),head_reference_error_mm=float(np.linalg.norm(actual[0]-points[0])*1000),maximum_local_correction_deg=float(np.degrees(abs(result.x-initial).max())),cost=float(result.cost),termination_status=int(result.status))
        return base.copy(),actual.copy(),robot.d.xmat[robot.head].reshape(3,3).copy()
