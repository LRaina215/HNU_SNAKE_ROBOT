"""Independent native FK, diagnostic registration only; no physical steps."""
from physics import *
from continuous_window_reference import ContinuousWindowReference
import json
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(INITIAL);d.qpos[:]=z['qpos'];mujoco.mj_forward(m,d)
qa=m.jnt_qposadr[m.actuator_trnid[:,0]];ref=ContinuousWindowReference(nodes(d),d.qpos[qa])
def fitted(native,C):
    x=native[10:];y=C[10:];cx=x.mean(0);cy=y.mean(0);u,sv,v=np.linalg.svd((x-cx).T@(y-cy));r=u@np.diag([1,1,np.linalg.det(u@v)])@v
    fit=(native-cx)@r+cy;e=np.linalg.norm(fit-C,axis=1)*1000
    return dict(rear_rms_mm=float(np.sqrt(np.mean(e[10:]**2))),front_rms_mm=float(np.sqrt(np.mean(e[:7]**2))),front_max_mm=float(max(e[:7])),full_rms_mm=float(np.sqrt(np.mean(e**2))))
if __name__=='__main__':
    rows=[]
    for t in [0,10,20,30,38,40,42,45,48,50,60,100,150,200,235]:
        q,i=ref.command(ref.feed_per_cycle*.02*max(0,t-.5));d.qpos[qa]=q;mujoco.mj_forward(m,d);row=dict(t=t,raw_deg=float(np.rad2deg(max(abs(q)))),guide_C_max_mm=i['guide_to_C_max_mm'],**fitted(nodes(d),i['nominal_curve_points']));rows.append(row)
    out=dict(physical_steps=0,registration='least squares rigid diagnostic using nodes10..21; never feeds command or curve',samples=rows)
    (BASE/'A10_NATIVE_STATIC.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
    
