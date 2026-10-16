"""Prescribed given coil, mirrored handedness and exit phase. No integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,mujoco,cv2
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;EXP=P.parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'
def forbidden(*a,**kw):raise RuntimeError('Given-pose setup has no time integration')
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,forbidden)
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);old=np.load(EXP/'STEP2_FROM_SOURCE_20261015/SOURCE_INITIAL.npz')
mujoco.mj_setState(m,d,old['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
origin=np.array([0.,0.,1.]);R=Rotation.from_quat(d.qpos[3:7][[1,2,3,0]]).as_matrix()
points=nodes(d);phi=np.arctan2(points[0,2]-1,points[0,1]);angle=np.radians(-90.)-phi;turn=Rotation.from_rotvec([angle,0,0]).as_matrix()
d.qpos[:3]=origin+turn@(d.qpos[:3]-origin);R=turn@R;d.qpos[3:7]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]];mujoco.mj_forward(m,d)
d.qpos[0]+=1.04-nodes(d)[0,0];d.qvel[:]=0.;d.time=0.;d.xfrc_applied[:]=0.;mujoco.mj_forward(m,d);points=nodes(d);centers=.5*(points[:-1]+points[1:]);head=m.body('head').id
tangent=(points[0]-points[1])/.0645;normal=np.array([np.cos(np.radians(-165.)),0.,np.sin(np.radians(-165.))]);entry_surface=np.array([1.1,.20,1.])+ .0565*normal;entry_center=np.array([1.1,.20,1.])+.0853*normal
def append_line(scene,a,b,color,r=.002,arrow=False):
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.r_[color,1.])
 mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,r,a,b);scene.ngeom+=1
def project(scene,p,w,h):
 cameras=scene.camera;pos=.5*(cameras[0].pos+cameras[1].pos);forward=.5*(cameras[0].forward+cameras[1].forward);up=.5*(cameras[0].up+cameras[1].up)
 forward/=np.linalg.norm(forward);up/=np.linalg.norm(up);right=np.cross(forward,up);v=p-pos;depth=v@forward;near=cameras[0].frustum_near;top=cameras[0].frustum_top;bottom=cameras[0].frustum_bottom
 x=(v@right)*near/depth;y=(v@up)*near/depth;width=(top-bottom)*w/h
 return (int(w*(.5+x/width)),int(h*(1-(y-bottom)/(top-bottom))))
m.vis.global_.offwidth=1400;m.vis.global_.offheight=1000
ren=mujoco.Renderer(m,width=1400,height=1000);cam=mujoco.MjvCamera();cam.distance=1.20;cam.lookat[:]=[.87,.055,1.]
images=[]
for az,el,label in [(90,-9,'SOURCE SIDE'),(140,-24,'OBLIQUE')]:
 cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam)
 for g in ren.scene.geoms:
  if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
   j=int(m.geom_bodyid[g.objid])-head
   if 0<=j<21:g.rgba[:3]=[1.,.35,.1] if j<8 else [.15,.65,1.]
 append_line(ren.scene,points[0],points[0]+.10*tangent,[1.,.9,.05],.004,True)
 append_line(ren.scene,entry_surface-[0,0,.08],entry_surface,[.1,1.,.25],.004,True)
 append_line(ren.scene,[.45,0,1],[.98,0,1],[.15,.8,1.],.001)
 im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
 cv2.putText(im,'PRESCRIBED INITIAL / NO DYNAMICS / '+label,(25,40),0,.8,(255,255,255),2)
 for point,text,color,offset in [(centers[0],'M0',(40,40,255),(-65,-25)),(points[0]+.10*tangent,'HEAD TANGENT',(40,240,255),(15,-10)),
  (centers[3],'RELEASE M0-M7',(50,150,255),(-200,60)),(centers[14],'RETAIN M8-M20',(255,190,60),(-280,-70)),(entry_surface,'RECEIVER ENTRY',(70,255,70),(20,30))]:
  xy=project(ren.scene,np.array(point),1400,1000);dest=(int(np.clip(xy[0]+offset[0],10,1050)),int(np.clip(xy[1]+offset[1],80,970)))
  cv2.circle(im,xy,7,color,-1);cv2.line(im,xy,dest,color,2);cv2.putText(im,text,dest,0,.65,color,2)
 images.append(im)
cv2.imwrite(str(P/'CANDIDATE_INITIAL_SIDE.png'),images[0]);cv2.imwrite(str(P/'CANDIDATE_INITIAL_OBLIQUE.png'),images[1]);cv2.imwrite(str(P/'CANDIDATE_INITIAL_DUAL.png'),np.concatenate(images,axis=1));ren.close()
components={};depth={}
for i,ct in enumerate(d.contact):
 a,b=ct.geom1,ct.geom2;ra=head<=m.geom_bodyid[a]<head+21;rb=head<=m.geom_bodyid[b]<head+21
 if ra==rb:continue
 name=m.geom(b if ra else a).name;f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);components[name]=components.get(name,0.)+max(0.,f[0]);depth[name]=max(depth.get(name,0.),-ct.dist)
state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(P/'CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy(),points=points,release_material_count=8)
report=dict(given_pose_not_historical_intermediate=True,integration_steps=0,handedness='Original given-coil chirality retained; prescribed source-axis placement and circumferential phase only',
 prescribed_head_x_m=1.04,prescribed_head_phase_deg=-90.,M0_center=centers[0].tolist(),M0_front_tip=points[0].tolist(),head_tangent=tangent.tolist(),entry_surface=entry_surface.tolist(),entry_center=entry_center.tolist(),
 source_center_radius_range_m=[float(np.linalg.norm(centers[:,1:]-[0,1],axis=1).min()),float(np.linalg.norm(centers[:,1:]-[0,1],axis=1).max())],
 source_x_center_range_m=[float(centers[:,0].min()),float(centers[:,0].max())],joint_peak_deg=float(np.max(abs(d.qpos[qa]))*180/np.pi),
 source_winding_deg=float(np.ptp(np.unwrap(np.arctan2(centers[:,2]-1,centers[:,1])))*180/np.pi),retained_M8to20_winding_deg=float(np.ptp(np.unwrap(np.arctan2(centers[8:,2]-1,centers[8:,1])))*180/np.pi),
 contact_normal_N=components,contact_max_depth_mm={k:v*1000 for k,v in depth.items()},qacc_max=float(abs(d.qacc).max()),status='Draft prescribed geometry only; static adjustment and fixed-source reach/clearance analysis pending')
(P/'CANDIDATE_INITIAL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
