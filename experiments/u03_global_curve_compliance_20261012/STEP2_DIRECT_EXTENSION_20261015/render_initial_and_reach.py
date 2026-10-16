"""Annotated original-model renders; every displayed pose is kinematic."""
from pathlib import Path
import sys,json,numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(ROOT))
from snake_control.suspended_supply_transfer import nodes
m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m);initial=np.load(P/'GIVEN_INITIAL.npz');path=np.load(P/'DIRECT_CONTACT_PATH.npz');head=m.body('head').id
m.vis.global_.offwidth=1400;m.vis.global_.offheight=1000;ren=mujoco.Renderer(m,width=1400,height=1000);cam=mujoco.MjvCamera();cam.distance=.97;cam.lookat[:]=[.91,.055,.98]
fontfile='C:/Windows/Fonts/msyh.ttc';font=ImageFont.truetype(fontfile,27);small=ImageFont.truetype(fontfile,22);title=ImageFont.truetype(fontfile,34)
normal=np.array([np.cos(np.radians(-165)),0.,np.sin(np.radians(-165))]);entry=np.array([1.1,.20,1.])+.0565*normal
def line(scene,a,b,color,r=.002,arrow=False):
 g=scene.geoms[scene.ngeom];kind=mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE;mujoco.mjv_initGeom(g,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.r_[color,1.]);mujoco.mjv_connector(g,kind,r,a,b);scene.ngeom+=1
def project(p):
 cameras=ren.scene.camera;pos=.5*(cameras[0].pos+cameras[1].pos);f=.5*(cameras[0].forward+cameras[1].forward);u=.5*(cameras[0].up+cameras[1].up);f/=np.linalg.norm(f);u/=np.linalg.norm(u);r=np.cross(f,u);v=p-pos
 near=cameras[0].frustum_near;top=cameras[0].frustum_top;bottom=cameras[0].frustum_bottom;z=v@f;width=(top-bottom)*1.4
 return np.array([1400*(.5+(v@r)*near/z/width),1000*(1-((v@u)*near/z-bottom)/(top-bottom))])
def label(draw,p,text,color,offset):
 xy=project(np.array(p));dest=xy+offset;dest[0]=np.clip(dest[0],20,1040);dest[1]=np.clip(dest[1],90,935)
 draw.line([tuple(xy),tuple(dest)],fill=color,width=2);draw.ellipse([xy[0]-6,xy[1]-6,xy[0]+6,xy[1]+6],fill=color,outline=(255,255,255),width=1)
 draw.multiline_text(tuple(dest),text,font=font,fill=color,stroke_width=1,stroke_fill=(15,20,24),spacing=5)
def render(pose,az,el,caption,show_path=False):
 d.qpos[:]=pose;mujoco.mj_kinematics(m,d);p=nodes(d);c=.5*(p[:-1]+p[1:]);T=(p[0]-p[1])/.0645;cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam)
 for i in range(ren.scene.ngeom):
  g=ren.scene.geoms[i]
  if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
   material=int(m.geom_bodyid[g.objid])-head
   if 0<=material<21:g.rgba[:3]=[1.,.34,.12] if material<8 else [.12,.65,1.]
   if material==0:g.rgba[:3]=[1.,.12,.10]
 line(ren.scene,p[0],p[0]+.08*T,[1.,.87,.05],.0035,True)
 line(ren.scene,entry-[0,0,.07],entry,[.15,1.,.35],.0035,True)
 if show_path:
  tips=path['points'][:,0]
  for a,b in zip(tips[:-1],tips[1:]):line(ren.scene,a,b,[1.,.88,.2],.0012)
 im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im);draw.rounded_rectangle((12,10,1388,78),radius=10,fill=(17,24,33));draw.text((30,23),caption,font=title,fill=(245,248,255))
 def fixed_label(point,text,color,destination):label(draw,point,text,color,np.array(destination)-project(np.array(point)))
 fixed_label(c[0],'M0',(255,70,60),(790,735));fixed_label(p[0]+.08*T,'头部切向',(255,226,65),(1015,820))
 fixed_label(c[3],'释放段 M0–M7\n长度 0.516 m',(255,154,75),(650,180));fixed_label(c[14],'保留源支撑 M8–M20\n前部释放期间不沿源轴前移',(78,193,255),(65,180))
 fixed_label(entry,'接收直管接入位置',(80,255,130),(1030,520));fixed_label([1.025,0,1],'源侧入弯段',(180,210,240),(1010,355))
 axis0=project(np.array([.60,0,1.]));axis1=project(np.array([.83,0,1.]));draw.line([tuple(axis0),tuple(axis1)],fill=(75,198,255),width=3);draw.text(tuple(axis0+[0,-38]),'源管轴 +X',font=small,fill=(90,205,255))
 draw.rounded_rectangle((18,905,1380,985),radius=8,fill=(17,24,33));draw.text((35,918),'给定姿态 / 运动学示意，不代表动力学已通过。',font=font,fill=(244,247,252));draw.text((35,955),'蓝=固定源段；橙=释放段；绿=接收位置。初态目标无接触。',font=small,fill=(195,210,226))
 return im
side=render(initial['qpos'],90,-10,'给定全源初态｜侧视｜M0 与释放/支撑分段');oblique=render(initial['qpos'],140,-24,'给定全源初态｜斜视｜未预接触接收管')
side.save(P/'GIVEN_INITIAL_SIDE.png');oblique.save(P/'GIVEN_INITIAL_OBLIQUE.png');dual=Image.new('RGB',(2800,1000));dual.paste(side,(0,0));dual.paste(oblique,(1400,0));dual.save(P/'GIVEN_INITIAL_DUAL.png')
panels=[]
for alpha,caption in [(0.,'α=0｜全源包绕给定初态'),(.5,'α=0.5｜前部伸出，源段保持'),(1.,'α=1｜从内侧下方向上首次接触')]:
 index=int(np.argmin(abs(path['alpha']-alpha)));panels.append(render(path['qpos'][index],140,-24,caption,True).resize((1120,800)))
sequence=Image.new('RGB',(3360,800))
for i,panel in enumerate(panels):sequence.paste(panel,(i*1120,0))
sequence.save(P/'DIRECT_EXTENSION_KINEMATIC_SEQUENCE.png');ren.close();print(json.dumps(dict(images=['GIVEN_INITIAL_SIDE.png','GIVEN_INITIAL_OBLIQUE.png','GIVEN_INITIAL_DUAL.png','DIRECT_EXTENSION_KINEMATIC_SEQUENCE.png'],source_data='GIVEN_INITIAL.npz + DIRECT_CONTACT_PATH.npz',dynamics=False)))
