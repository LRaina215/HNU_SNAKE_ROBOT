"""Saved actual states only, continuous 1x, side/oblique; no physical replay."""
from physics import *
import argparse,json,cv2
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();case=BASE/a.case
plan=json.loads((case/'plan.json').read_text());shift_unit='cells' if (plan['parameters'].get('two_windows') or plan['parameters'].get('continuous_windows') or plan['parameters'].get('supply_windows') or plan['parameters'].get('handover_windows') or plan['parameters'].get('overlap_windows')) else 'm';z=np.load(case/'trace.npz');m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);ren=mujoco.Renderer(m,width=700,height=520);cam=mujoco.MjvCamera();cam.lookat[:]=[.95,.34,1];cam.distance=1.6
wr=cv2.VideoWriter(str(case/'actual_side_oblique_1x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1400,520));tt=z['t'];shots=[0,10,20,30,35,40,50,60,80,100,120,140,160,180,200,220,234];saved={}
for t in np.arange(tt[0],tt[-1]+.001,.05):
    i=min(np.searchsorted(tt,t),len(tt)-1);d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);lower=np.clip((.6-z['points'][i,:,2].min())/.6,0,1);cam.lookat[2]=1-.45*lower;cam.distance=1.6+.45*lower;ims=[]
    for az,el,label in [(0,0,'SIDE Y horizontal / Z up'),(140,-26,'OBLIQUE')]:
        cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam);im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR);cv2.putText(im,f'{a.case} saved ACTUAL 1x {label}',(10,25),0,.51,(255,255,255),1);cv2.putText(im,f't={tt[i]:.1f}s reference S={z["S"][i]:.3f}{shift_unit}',(10,47),0,.55,(255,255,255),1);cv2.putText(im,f'normal sums source={z["forces"][i,0].sum():.1f} target={z["forces"][i,1].sum():.1f} N',(10,68),0,.5,(255,255,255),1);ims.append(im)
    im=np.concatenate(ims,axis=1);wr.write(im);sec=int(round(t))
    if sec in shots and sec not in saved:cv2.imwrite(str(case/f'side_{sec:03d}s.png'),im);saved[sec]=True
    if int(round(t*20))%800==0:print('render',round(t,1),flush=True)
cv2.imwrite(str(case/'actual_final_frame.png'),im)
wr.release();ren.close();cap=cv2.VideoCapture(str(case/'actual_side_oblique_1x.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);checks=[]
for fi in [0,count//2,count-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,fi);checks.append(bool(cap.read()[0]))
cap.release();(case/'video_metadata.json').write_text(json.dumps(dict(fps=fps,frames=count,duration=count/fps,readback=checks,source='saved actual qpos at0.1s, repeated to20fps; no interpolation or dynamics rerun',camera_only_adapts_to_actual_height=True),indent=2))


