"""Annotated prescribed initial only; original scene, no integration."""
import argparse,json
from PIL import Image,ImageDraw,ImageFont
from physics import *

def run(args):
    out=Path(args.out).resolve();out.mkdir(exist_ok=True);z=np.load(args.initial)
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);d.qpos[:]=z['qpos'];mujoco.mj_kinematics(m,d)
    W,H=1400,1000;m.vis.global_.offwidth=W;m.vis.global_.offheight=H;renderer=mujoco.Renderer(m,width=W,height=H)
    camera=mujoco.MjvCamera();camera.distance=1.02;camera.lookat[:]=[.91,.06,.98];head=m.body('head').id;p=nodes(d);centers=.5*(p[:-1]+p[1:]);T=(p[0]-p[1])/.0645
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',32)
    entry=np.array([1.1,.20,1.])+.0565*np.array([np.cos(np.radians(-165)),0.,np.sin(np.radians(-165))])
    def connector(a,b,color,arrow=False):
        scene=renderer.scene;g=scene.geoms[scene.ngeom];kind=mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
        mujoco.mjv_initGeom(g,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.r_[color,1.]);mujoco.mjv_connector(g,kind,.003,a,b);scene.ngeom+=1
    def project(point):
        c=renderer.scene.camera;position=.5*(c[0].pos+c[1].pos);forward=.5*(c[0].forward+c[1].forward);up=.5*(c[0].up+c[1].up);forward/=np.linalg.norm(forward);up/=np.linalg.norm(up);right=np.cross(forward,up);v=point-position
        near=c[0].frustum_near;top=c[0].frustum_top;bottom=c[0].frustum_bottom;zv=v@forward
        return np.array([W*(.5+(v@right)*near/zv/((top-bottom)*W/H)),H*(1-((v@up)*near/zv-bottom)/(top-bottom))])
    def render(az,el,caption):
        camera.azimuth=az;camera.elevation=el;renderer.update_scene(d,camera)
        for g in renderer.scene.geoms[:renderer.scene.ngeom]:
            if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
                i=int(m.geom_bodyid[g.objid])-head
                if 0<=i<21:g.rgba[:3]=[1.,.34,.12] if i<8 else [.12,.65,1.]
                if i==0:g.rgba[:3]=[1.,.12,.10]
        connector(p[0],p[0]+.085*T,[1.,.87,.05],True);connector(entry-[0,0,.075],entry,[.15,1.,.35],True)
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((12,10,1388,80),10,fill=(17,24,33));draw.text((30,24),caption,font=title,fill=(245,248,255))
        def label(point,text,color,destination):
            xy=project(np.asarray(point));dest=np.array(destination);draw.line([tuple(xy),tuple(dest)],fill=color,width=2);draw.ellipse([xy[0]-5,xy[1]-5,xy[0]+5,xy[1]+5],fill=color,outline=(255,255,255));draw.multiline_text(tuple(dest),text,font=font,fill=color,stroke_width=1,stroke_fill=(15,20,24),spacing=5)
        label(centers[0],'M0',(255,75,65),(800,725));label(p[0]+.085*T,'头部切向：向 +Y、向上',(255,226,65),(965,820))
        label(centers[3],'前部释放 M0–M7\n0.516 m；初态较松',(255,154,75),(650,140))
        label(centers[14],'保留源支撑 M8–M20\n0.8385 m；不预先送进/滚动',(78,193,255),(55,170))
        label(entry,'目标接入位置',(80,255,130),(1040,505))
        a=project(np.array([.60,0,1.]));b=project(np.array([.83,0,1.]));draw.line([tuple(a),tuple(b)],fill=(75,198,255),width=3);draw.text(tuple(a+[0,-38]),'源管轴 +X',font=small,fill=(90,205,255))
        draw.rounded_rectangle((18,902,1380,987),8,fill=(17,24,33));draw.text((35,914),'未审查候选｜21杆均在源管侧，目标无接触；不是已伸出/已包绕的中途状态。',font=font,fill=(244,247,252));draw.text((35,955),'原模型静态预载；未运行动力学。前部中心距源轴较大，需与紧贴管初态区别。',font=small,fill=(195,210,226))
        return image
    side=render(90,-10,'V7 给定全源候选初态｜侧视');oblique=render(140,-24,'V7 给定全源候选初态｜斜视')
    side.save(out/'INITIAL_SIDE.png');oblique.save(out/'INITIAL_OBLIQUE.png');dual=Image.new('RGB',(2800,1000));dual.paste(side,(0,0));dual.paste(oblique,(1400,0));dual.save(out/'INITIAL_DUAL.png');renderer.close()
    print(json.dumps(dict(initial=args.initial,images=['INITIAL_SIDE.png','INITIAL_OBLIQUE.png','INITIAL_DUAL.png'],simulation_steps=0)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--out',required=True);run(ap.parse_args())
