"""Offline label check only; labels never enter the controller."""
import argparse,json
from physics import *
from contact_sensor import read_sensor,known_source
ap=argparse.ArgumentParser();ap.add_argument('initial');args=ap.parse_args();reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
for ct in read_sensor(m,d)['contacts']:
    if not known_source(ct,reg['source_known']):
        p=ct['point'];print(json.dumps(dict(material=ct['material'],point=p.tolist(),normal=ct['normal'].tolist(),source_radial_error_m=float(np.linalg.norm(p[1:]-[0,1])-.0565),source_miter_error_m=float(p[0]+p[1]-1.1),normal_N=ct['normal_N'])))
