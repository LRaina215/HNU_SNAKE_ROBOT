"""Forward-only candidate load report; no integration or acceptance claims."""
import argparse,json
from physics import *
from contact_sensor import read_sensor,known_source
ap=argparse.ArgumentParser();ap.add_argument('initial');args=ap.parse_args();reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);sensor=read_sensor(m,d)
source=[ct for ct in sensor['contacts'] if known_source(ct,reg['source_known'])];other=[ct for ct in sensor['contacts'] if not known_source(ct,reg['source_known'])]
print(json.dumps(dict(initial=args.initial,source_vertical_N=float(sum(ct['world_force_N'][2] for ct in source)),source_normal_N=float(sum(ct['normal_N'] for ct in source)),non_source_contacts=len(other),torque_peak_Nm=float(abs(d.actuator_force).max()),qacc_peak=float(abs(d.qacc).max()),root_qacc=d.qacc[:6].tolist(),simulation_steps=0),indent=2))
