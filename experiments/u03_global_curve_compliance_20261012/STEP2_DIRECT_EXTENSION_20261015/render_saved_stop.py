"""Render a preserved final physical state without integrating or editing it."""
from pathlib import Path
import argparse,json,numpy as np,mujoco,cv2
ap=argparse.ArgumentParser();ap.add_argument('case_directory');args=ap.parse_args();C=Path(args.case_directory).resolve();m=mujoco.MjModel.from_xml_path(str(C/'model.xml'));d=mujoco.MjData(m);state=np.load(C/'integration_state.npz');mujoco.mj_setState(m,d,state['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.9;cam.lookat[:]=[.94,.25,1.];images=[]
for az,el,label in [(90,-8,'SOURCE SIDE'),(140,-26,'OBLIQUE')]:
    cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);frame=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(frame,f'ACTUAL SAVED STOP {label} t={d.time:.3f}s',(10,25),0,.45,(255,255,255),1);images.append(frame)
cv2.imwrite(str(C/'actual_exact_stop.png'),np.concatenate(images,axis=1));ren.close();print(str(C/'actual_exact_stop.png'))
