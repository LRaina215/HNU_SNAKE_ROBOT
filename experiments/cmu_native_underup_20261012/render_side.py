"""Replay actual saved states in fixed side and oblique views at 1x."""
from native_run import *
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--source');a=ap.parse_args();case=BASE/a.case;case.mkdir(exist_ok=True);source=Path(a.source) if a.source else case
z=np.load(source/'trace.npz');m=mujoco.MjModel.from_xml_path(str(source/'model.xml'));d=mujoco.MjData(m)
ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.distance=1.6;cam.lookat[:]=[.95,.34,1.0]
wr=cv2.VideoWriter(str(case/'actual_side_oblique_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1400,520))
shots=[0,20,40,50,60,70,90,110,130,150,180,210,234];saved={};t=z['t']
for ti in np.arange(t[0],t[-1]+.001,.05):
    i=min(np.searchsorted(t,ti),len(t)-1);d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);ims=[]
    for az,el,label in [(0,0,'SIDE looking along X; Y horizontal / Z upward'),(140,-26,'OBLIQUE')]:
        cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(im,label,(12,26),0,.5,(255,255,255),1);cv2.putText(im,f'{a.case} ACTUAL t={t[i]:.1f}s replay 1x',(12,49),0,.56,(255,255,255),1);ims.append(im)
    im=np.concatenate(ims,axis=1);wr.write(im);sec=int(round(ti))
    if sec in shots and sec not in saved:cv2.imwrite(str(case/f'side_{sec:03d}s.png'),im);saved[sec]=True
    if int(round(ti*20))%800==0:print('render',round(ti,1),flush=True)
wr.release();ren.close();cap=cv2.VideoCapture(str(case/'actual_side_oblique_1x.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);reads=[]
for i in [0,count//2,count-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);reads.append(bool(cap.read()[0]))
cap.release();(case/'video_metadata.json').write_text(json.dumps(dict(frames=count,fps=fps,duration=count/fps,readable_sampled_frames=reads,source='saved actual qpos, 0.1s samples repeated to 20fps; no physics rerun',camera='fixed world side and oblique'),indent=2))
frames=[cv2.resize(cv2.imread(str(case/f'side_{s:03d}s.png')),(700,260)) for s in shots if s in saved]
cv2.imwrite(str(case/'side_contact_sheet.png'),np.concatenate(frames,axis=0))
