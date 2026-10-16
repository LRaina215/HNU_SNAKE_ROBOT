"""Render saved real qpos only, at 1x, with a closer following camera."""
from native_run import *
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);args=ap.parse_args();case=BASE/args.case
z=np.load(case/'trace.npz');m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m)
ren=mujoco.Renderer(m,width=1000,height=600);cam=mujoco.MjvCamera();cam.distance=1.25;cam.azimuth=140;cam.elevation=-26
wr=cv2.VideoWriter(str(case/'actual_continuous_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1000,600));start=float(z['t'][0]);end=float(z['t'][-1]);last_center=z['centers'][0].mean(0)
shots={0,25,50,65,90,115,155,185,234};saved=set()
for ti in np.arange(start,end+.001,.05):
    i=min(int(round((ti-start)/.1)),len(z['t'])-1);d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d)
    last_center=.97*last_center+.03*z['centers'][i].mean(0);cam.lookat[:]=last_center;ren.update_scene(d,cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
    cv2.putText(im,f'{args.case} saved ACTUAL dynamics, continuous 1x / t={z["t"][i]:.1f}s',(15,28),0,.65,(255,255,255),1)
    cv2.putText(im,f'nt={z["nt"][i]:.2f} L={z["L"][i]:.2f} source={z["forces"][i,0].sum():.1f}N target={z["forces"][i,1].sum():.1f}N',(15,52),0,.58,(255,255,255),1)
    wr.write(im)
    sec=int(round(ti))
    if sec in shots and sec not in saved:cv2.imwrite(str(case/f'actual_{sec:03d}s.png'),im);saved.add(sec)
    if round(ti*20)%600==0:print('render',round(ti,1),flush=True)
wr.release();ren.close();cap=cv2.VideoCapture(str(case/'actual_continuous_1x.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);checks=[]
for fi in [0,count//2,count-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,_=cap.read();checks.append(dict(frame=fi,readable=bool(ok)))
cap.release();(case/'video_1x.json').write_text(json.dumps(dict(fps=fps,frames=count,duration=count/fps,source='saved actual qpos at .1s; nearest-frame replay to20fps, no interpolation',physical_state_modified=False,readback=checks),indent=2))
frames=[cv2.resize(cv2.imread(str(case/f'actual_{sec:03d}s.png')),(500,300)) for sec in sorted(shots) if (case/f'actual_{sec:03d}s.png').exists()]
cv2.imwrite(str(case/'close_contact_sheet.png'),np.concatenate([np.concatenate(frames[i:i+3],1) for i in range(0,9,3)],0))
