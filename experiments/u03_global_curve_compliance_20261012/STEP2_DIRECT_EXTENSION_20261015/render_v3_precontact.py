"""Render archived ideal-C poses; explicit kinematic labels, no integration."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V3_POSITIVE_POLAR';O=B/'PRECONTACT_EVALUATION';sys.path.insert(0,str(B))
from physics import MODEL,nodes
from rough_planner import CurvatureCoordinates
z=np.load(O/'PRECONTACT_PREFIX.npz');m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);qa=m.jnt_qposadr[m.actuator_trnid[:,0]];head=m.body('head').id;coords=CurvatureCoordinates()
fields=np.einsum('nd,abd->nab',z['qpos'][:,qa],coords.modes)
metrics=dict(simulation_steps=0,geometric_native_joint_peak_deg=float(abs(z['qpos'][:,qa]).max()*180/np.pi),
    geometric_curvature_peak_per_m=float(np.linalg.norm(fields,axis=2).max()*1000),native_integral_error_rad=float(abs(np.einsum('ja,nab->njb',coords.A,fields)[:,:,0][:,np.arange(20)%2==1]+z['qpos'][:,qa][:,np.arange(20)%2==1]).max()))
# Check both alternating native components, not a reconstructed Bishop curve.
reconstructed=np.array([coords.hinges(field) for field in fields]);metrics['native_integral_error_rad']=float(abs(reconstructed-z['qpos'][:,qa]).max())
W,H=1120,800;m.vis.global_.offwidth=W;m.vis.global_.offheight=H;renderer=mujoco.Renderer(m,width=W,height=H);camera=mujoco.MjvCamera();camera.distance=1.04;camera.lookat[:]=[.91,.055,.98];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28)
def project(point):
    c=renderer.scene.camera;position=.5*(c[0].pos+c[1].pos);forward=.5*(c[0].forward+c[1].forward);up=.5*(c[0].up+c[1].up);forward/=np.linalg.norm(forward);up/=np.linalg.norm(up);right=np.cross(forward,up);v=point-position;near=c[0].frustum_near;top=c[0].frustum_top;bottom=c[0].frustum_bottom
    return np.array([W*(.5+(v@right)*near/(v@forward)/((top-bottom)*W/H)),H*(1-((v@up)*near/(v@forward)-bottom)/(top-bottom))])
def render(index,az,el):
    d.qpos[:]=z['qpos'][index];mujoco.mj_kinematics(m,d);p=nodes(d);centers=.5*(p[:-1]+p[1:]);T=(p[0]-p[1])/.0645;camera.azimuth=az;camera.elevation=el;renderer.update_scene(d,camera)
    for g in renderer.scene.geoms[:renderer.scene.ngeom]:
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
            material=int(m.geom_bodyid[g.objid])-head
            if 0<=material<21:g.rgba[:3]=[1.,.34,.12] if material<8 else [.12,.65,1.]
            if material==0:g.rgba[:3]=[1.,.12,.10]
    image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rounded_rectangle((10,8,W-10,67),8,fill=(17,24,33));draw.text((24,22),f'运动学理想跟随｜t={z["t"][index]:.3f}s  α={z["alpha"][index]:.3f}',font=title,fill='white')
    tip=project(p[0]);end=project(p[0]+.075*T);draw.line([tuple(tip),tuple(end)],fill=(255,226,65),width=4);v=(end-tip)/max(np.linalg.norm(end-tip),1e-9);normal=np.array([-v[1],v[0]])
    draw.polygon([tuple(end),tuple(end-13*v+6*normal),tuple(end-13*v-6*normal)],fill=(255,226,65));xy=project(centers[0]);draw.ellipse((xy[0]-5,xy[1]-5,xy[0]+5,xy[1]+5),fill=(255,60,50));draw.text(tuple(xy+[12,5]),'M0',font=font,fill=(255,70,60),stroke_width=1,stroke_fill='black')
    draw.rounded_rectangle((10,H-88,W-10,H-10),8,fill=(17,24,33));draw.text((24,H-78),'橙：释放前8杆；蓝：固定M8–M20；黄：头部切向。',font=font,fill='white');draw.text((24,H-43),'0动力学步；不代表真实源保持、载荷或接管已经通过。',font=font,fill=(215,225,235))
    return image
panels=[render(i,140,-24) for i in [0,len(z['t'])//2,len(z['t'])-1]];sequence=Image.new('RGB',(3*W,H))
for j,panel in enumerate(panels):sequence.paste(panel,(j*W,0))
sequence.save(O/'KINEMATIC_RELEASE_SEQUENCE.png');render(len(z['t'])-1,90,-10).save(O/'KINEMATIC_FIRST_CONTACT_SIDE.png');renderer.close()
(O/'NATIVE_MAPPING_METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
