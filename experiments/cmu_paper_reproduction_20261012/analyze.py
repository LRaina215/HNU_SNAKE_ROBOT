from pathlib import Path
import numpy as np,cv2,json,sys
from run import cylinder
BASE=Path(__file__).resolve().parent
for name in sys.argv[1:]:
    case=BASE/name;z=np.load(case/'trace.npz');v=cv2.VideoCapture(str(case/'actual_continuous.mp4'));frames=[]
    report=[]
    times=np.linspace(0,z['t'][-1],8)
    for t in times:
        i=np.argmin(abs(z['t']-t));c=z['centers'][i];f=z['forces'][i]
        err=abs(z['ctrl'][i]-z['qpos'][i,7:]);rr=np.linalg.norm(c[:,[0,2]]-[0,1],axis=1)
        report.append(dict(t=float(z['t'][i]),nt=float(z['nt'][i]),head=c[:5].tolist(),source=np.flatnonzero(f[0]>.1).tolist(),target=np.flatnonzero(f[1]>.1).tolist(),tracking_deg=np.rad2deg(err).tolist(),target_radius=rr.tolist()))
        v.set(cv2.CAP_PROP_POS_MSEC,t*.5*1000);ok,frame=v.read()
        if ok:
            cv2.putText(frame,f't={t:.1f}s',(10,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2);frames.append(cv2.resize(frame,(450,300)))
    v.release()
    while len(frames)<8:frames.append(frames[-1])
    cv2.imwrite(str(case/'contact_sheet.png'),np.concatenate([np.concatenate(frames[:4],1),np.concatenate(frames[4:8],1)],0))
    (case/'analysis_samples.json').write_text(json.dumps(report,indent=2))
    a,o,r,p,e=cylinder(z['centers'][-1]);print(name,'final helix',a,o,r,p,e)
    for r in report:print(round(r['t'],1),'src',r['source'],'tgt',r['target'],'head',np.round(r['head'][0],3),'error',np.round(r['tracking_deg'],1))
