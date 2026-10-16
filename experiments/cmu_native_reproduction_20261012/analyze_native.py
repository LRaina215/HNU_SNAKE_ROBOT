from native_run import *
for name in sys.argv[1:]:
    case=BASE/name;z=np.load(case/'trace.npz');v=cv2.VideoCapture(str(case/'actual_continuous.mp4'));frames=[];report=[]
    for t in np.linspace(z['t'][0],z['t'][-1],8):
        i=np.argmin(abs(z['t']-t));c=z['centers'][i];f=z['forces'][i];rr=np.linalg.norm(c[:,[0,2]]-[1.1,1],axis=1)
        s=cylinder(z['points'][i]);phi=np.unwrap(np.arctan2(c[:8,2]-1,c[:8,0]-1.1));q=z['qpos'][i,7:]
        report.append(dict(t=float(z['t'][i]),nt=float(z['nt'][i]),head=z['points'][i,0].tolist(),target_radius=rr.tolist(),source_modules=np.flatnonzero(f[0]>.1).tolist(),target_modules=np.flatnonzero(f[1]>.1).tolist(),front_span_deg=float(np.rad2deg(np.ptp(phi))),front_target_force=float(f[1,:8].sum()),cylinder={k:(val.tolist() if isinstance(val,np.ndarray) else val) for k,val in s.items()},tracking_deg=np.rad2deg(abs(z['ctrl'][i]-q)).tolist()))
        v.set(cv2.CAP_PROP_POS_MSEC,(t-z['t'][0])*.5*1000);ok,im=v.read()
        if ok:
            cv2.putText(im,f't={t:.1f}s',(10,65),0,.8,(255,255,255),2);frames.append(cv2.resize(im,(450,300)))
    v.release()
    while len(frames)<8:frames.append(frames[-1])
    cv2.imwrite(str(case/'contact_sheet.png'),np.concatenate([np.concatenate(frames[:4],1),np.concatenate(frames[4:8],1)],0))
    (case/'analysis_samples.json').write_text(json.dumps(report,indent=2))
    # Entire-body acceptance is physical, not the controller nt or just a contact.
    r=np.linalg.norm(z['centers'][:,:,[0,2]]-[1.1,1],axis=2)
    mask=np.all((r>.07)&(r<.12),axis=1)&np.all((z['centers'][:,:,1]>.15)&(z['centers'][:,:,1]<1.2),axis=1)&(z['forces'][:,0].sum(1)<.1)&(z['forces'][:,1].sum(1)>.2)
    bands=np.flatnonzero(mask)
    analysis=dict(any_full_target_geometry_and_force=bool(len(bands)),full_target_samples=len(bands),first_full_sample=None if not len(bands) else float(z['t'][bands[0]]),actual_carried_M18_displacement=(z['centers'][-1,18]-z['centers'][0,18]).tolist(),front_never_confirmed_by_contact_alone=True)
    (case/'acceptance_analysis.json').write_text(json.dumps(analysis,indent=2))
    print(name,analysis)
    for r in report:print(round(r['t'],1),'nt',round(r['nt'],2),'src',r['source_modules'],'tgt',r['target_modules'],'head',np.round(r['head'],3),'frontN',round(r['front_target_force'],1),'frontspan',round(r['front_span_deg'],1))
