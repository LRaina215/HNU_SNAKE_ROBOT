from native_run import *
ap=argparse.ArgumentParser();ap.add_argument('--nb',type=int,default=6);ap.add_argument('--seam',type=float,default=.84);args=ap.parse_args()
m=mujoco.MjModel.from_xml_path(str(MODEL));records=[]
for phase in np.linspace(-np.pi,np.pi,73)[:-1]:
    d,s=placement(m,1.2351672327411283,.19063617553842116,phase,args.nb,args.seam,0)
    p=nodes(d);tangent=(p[args.nb]-p[args.nb+1])/LM
    polar=np.arctan2(tangent[1],tangent[2])
    if np.cos(polar)<0:continue
    d,s=placement(m,1.2351672327411283,.19063617553842116,phase,args.nb,args.seam,polar)
    row,pairs=snapshot(m,d);tangent=(row['points'][args.nb]-row['points'][args.nb+1])/LM
    records.append(dict(phase=float(phase),polar=float(polar),tangent=tangent.tolist(),depth=row['depth'].tolist(),head=row['points'][0].tolist(),selected_center=row['centers'][args.nb].tolist(),cost=float(abs(tangent[0])+20*max(row['depth'][:3]))))
records.sort(key=lambda x:x['cost']);(BASE/f'native_trigger_scan_nb{args.nb}_x{args.seam}.json').write_text(json.dumps(records,indent=2))
for r in records[:8]:print(r)
