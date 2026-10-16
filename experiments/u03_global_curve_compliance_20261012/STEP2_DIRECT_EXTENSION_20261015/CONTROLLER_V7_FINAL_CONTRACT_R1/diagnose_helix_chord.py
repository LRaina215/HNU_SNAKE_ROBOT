"""Known head collider versus rough helix chord geometry, without integration."""
import json
from physics import *
from geometry import Geometry,normalize
from rough_planner import coarse_model
from scipy.optimize import brentq
g=Geometry([1.1,.2,1.],[0.,1.,0.],.0565);m=coarse_model(g);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);head=m.geom('head_shell').id;target=m.geom('rough_receiver').id;pitch=.18266863819422827
for radius in [.091,.0955,.09918695,.102,.105]:
 rate=1/np.sqrt(radius**2+(pitch/(2*np.pi))**2);a=pitch*rate/(2*np.pi);span=brentq(lambda h:(a*h)**2+4*radius**2*np.sin(rate*h/2)**2-.0645**2,.0645,.129)
 p0=np.array([1.1+radius,.6,1.]);p1=np.array([1.1+radius*np.cos(rate*span),.6-a*span,1+radius*np.sin(rate*span)]);T=normalize(p0-p1);N=normalize(np.cross([0,1,0],T));B=np.cross(T,N);R=np.column_stack([N,B,T]);d.geom_xpos[head]=.5*(p0+p1);d.geom_xmat[head]=R.ravel();gap=mujoco.mj_geomDistance(m,d,head,target,.1,None)
 print(json.dumps(dict(radius=radius,head_collider_gap_mm=float(gap*1000),step_deg=float(rate*span*180/np.pi),six_centers_span_deg=float(5*rate*span*180/np.pi))))
