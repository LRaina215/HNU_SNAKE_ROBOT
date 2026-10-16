from pathlib import Path
import mujoco,xml.etree.ElementTree as E,json,numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
for name in ['Snake_Robot_o20_Pipe_elbow90_horizontal.xml','Snake_Robot_o20_full_source_inner.xml']:
    p=ROOT/'mujoco_model'/name;r=E.parse(p).getroot();m=mujoco.MjModel.from_xml_path(str(p))
    h=m.body('head').id
    result=dict(path=str(p),mass=float(m.body_mass.sum()),force_range=m.actuator_forcerange.tolist(),gain=m.actuator_gainprm[:,:3].tolist(),bias=m.actuator_biasprm[:,:3].tolist(),dt=m.opt.timestep,joint_range=m.jnt_range[1:].tolist(),joint_axes=m.jnt_axis[1:].tolist(),robot_friction=m.geom_friction[m.geom_bodyid>=h].tolist(),pipe_geometries=[g.attrib for g in r.findall('./worldbody/geom') if g.get('name','').startswith('pipe_')],head=r.find('./worldbody/body[@name="head"]').attrib)
    (BASE/(name+'.inspection.json')).write_text(json.dumps(result,indent=2))
    print(name,'mass',result['mass'],'force',result['force_range'][:1],'gain',result['gain'][:1],'bias',result['bias'][:1],'dt',result['dt'],'joint_range',result['joint_range'][:2],'axes',result['joint_axes'][:4],'robot_friction',result['robot_friction'][:2])
    print('robot mass',m.body_mass[h:].sum(),'head',result['head'])
    print('pipes',[(g.attrib) for g in r.iter('geom') if g.get('name','').startswith('pipe_')])
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    print('world pipes',[(m.geom(i).name,d.geom_xpos[i].tolist(),d.geom_xmat[i].reshape(3,3)[:,2].tolist(),m.geom_size[i].tolist()) for i in range(m.ngeom) if (m.geom(i).name or '').startswith('pipe_')])
