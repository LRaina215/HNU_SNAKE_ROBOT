"""Read-only saved-contact identifiability and continuity checks before freezing."""
from physics import *
import json
from contact_sensor import read_sensor
from cylinder_estimator import CylinderEstimator
from geometry import Geometry
from global_update import probe_command,GlobalUpdater
from estimated_curve import EstimatedCurve

registration=json.loads((BASE/'REGISTERED.json').read_text());constants=json.loads((BASE/'source_constants.json').read_text());z=np.load(BASE.parent/'STEP1_20261014/N_SHAPE/trace.npz')
m=mujoco.MjModel.from_xml_path(str(BASE/'GIVEN_INITIAL/model.xml'));d=mujoco.MjData(m);results=[]
for condition,prior in registration['conditions'].items():
    geometry=Geometry.from_dict(prior);est=CylinderEstimator(geometry,registration['source_known'],registration['estimator'])
    for i in np.flatnonzero((z['t']>=90.-1e-6)&(z['t']<=112.+1e-6)):
        t=float(z['t'][i]-90.);d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i];mujoco.mj_forward(m,d)
        sensor=read_sensor(m,d);probe=probe_command(t,est.first_contact,registration['probe']);est.observe(sensor,t,probe)
    results.append(dict(condition=condition,first_contact=est.first_contact,first_reliable=est.first_reliable,last_fit=est.last,
        scope='Existing accurate-geometry trajectory contacts; no new dynamics and no formal outcome. Probe signals mark timing only, do not imply physical probing of saved trajectory.'))
    ref=EstimatedCurve(constants,geometry);ref.command(float(z['S'][np.argmin(abs(z['t']-90.))]));up=GlobalUpdater(registration['global_update'],True)
    if est.last['reliable']:
        up.update(ref,est,14.2,23.);results[-1]['private_first_update']=up.history[-1]
out=dict(saved_contact_checks=results,controller_input_permission_review='Controller files read no environment geom names, mesh vertices, target axis/radius or target labels; known source quantities remain explicit',new_dynamics_runs=0)
(BASE/'OBSERVATION_PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
