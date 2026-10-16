from run import *
def distances(m,d):
    result={'source':[],'target':[],'self':[],'floor':[]}
    for c in d.contact:
        names=[m.geom(g).name for g in [c.geom1,c.geom2]]
        kind='self' if all(s.startswith('robot') for s in names) else next((s for s in names if s in result),None)
        if kind:result[kind].append(dict(geoms=names,distance=float(c.dist)))
    return {key:dict(min_distance=min((x['distance'] for x in val),default=0),contacts=val) for key,val in result.items()}
if __name__=='__main__':
    for name in sys.argv[1:]:
        case=BASE/name;m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);z=np.load(case/'initial.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d)
        result=distances(m,d);(case/'saved_initial_contact_audit.json').write_text(json.dumps(result,indent=2));print(name,{k:v['min_distance'] for k,v in result.items()})
    m=mujoco.MjModel.from_xml_string(make_model())
    for off in [-.14,-.15,-.16,-.17,-.18]:
        d,s=initial(m,1.42,.23089076,0,4,off,0);print('offset',off,{k:v['min_distance'] for k,v in distances(m,d).items()})
