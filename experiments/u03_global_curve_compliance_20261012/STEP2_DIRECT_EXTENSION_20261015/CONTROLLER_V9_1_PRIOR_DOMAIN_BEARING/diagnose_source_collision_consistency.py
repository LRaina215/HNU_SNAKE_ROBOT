"""Compare only known-source collision queries at an archived unsafe pose."""
import argparse,json
from physics import *
from geometry import Geometry
from rough_planner import RoughPlanner

ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--trace',required=True);args=ap.parse_args()
reg=json.loads((BASE/'REGISTERED.json').read_text());m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial)
mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
plan=RoughPlanner(Geometry.from_dict(reg['conditions']['DIRECTION']),d.qpos.copy(),d.ctrl.copy(),reg['kinematic_solver']);trace=np.load(args.trace);d.qpos[:]=trace['qpos'][-1];mujoco.mj_fwdPosition(m,d);plan.observe(d.qpos.copy(),14);plan.fk(d.qpos[plan.qa]);gap,point,normal=plan.legacy_source_contacts(21)
actual=[];h=m.body('head').id
for ct in d.contact:
    gs=[int(ct.geom1),int(ct.geom2)];robot=[h<=m.geom_bodyid[g]<h+21 for g in gs]
    if sum(robot)!=1:continue
    j=int(robot[1]);name=m.geom(gs[1-j]).name
    if name in ['pipe_straight_01','pipe_sharp_in_02']:
        i=int(m.geom_bodyid[gs[j]])-h;k=0 if name=='pipe_straight_01' else 1
        actual.append(dict(material=i,source=name,actual_depth_mm=-float(ct.dist)*1000,private_depth_mm=-float(gap[i,k])*1000))
print(json.dumps(dict(simulation_steps=0,pose_max_difference=float(abs(d.qpos-plan.d.qpos).max()),known_source_contact_comparison=actual),indent=2))
