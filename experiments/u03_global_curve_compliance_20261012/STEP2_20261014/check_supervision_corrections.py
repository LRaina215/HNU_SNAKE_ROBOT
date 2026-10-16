"""Necessary receiver separation and all-proposal-rejection checks, no steps."""
from physics import *
import json
from geometry import Geometry
from contact_sensor import read_sensor,classified_observation
from estimated_observer import EstimatedObserver
from estimated_curve import EstimatedCurve
from global_update import GlobalUpdater
from types import SimpleNamespace
reg=json.loads((BASE/'REGISTERED.json').read_text());constants=json.loads((BASE/'source_constants.json').read_text())
m=mujoco.MjModel.from_xml_path(str(BASE/'GIVEN_INITIAL/model.xml'));d=mujoco.MjData(m);z=np.load(BASE/'GIVEN_INITIAL/integration_state.npz')
mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);sensor=read_sensor(m,d);rows=[]
for name,prior in reg['conditions'].items():
    geometry=Geometry.from_dict(prior);observed=classified_observation(sensor,geometry,reg['source_known'])
    rows.append(dict(condition=name,source_receiver_unknown_force_N=observed['forces'].sum(1).tolist(),contact_classes=observed['contact_estimated_classes']))
    assert np.isclose(observed['forces'].sum(),sum(x['normal_N'] for x in sensor['contacts']))
g=Geometry.from_dict(reg['conditions']['POSITION_RADIUS']);e,_=g.frame();ambiguous=dict(material=5,point=g.entry+g.radius*e,normal=g.axis.copy(),normal_N=12.,penetration_m=.0001)
synthetic=dict(sensor,contacts=[ambiguous],centers=np.tile(g.entry+(g.radius+.036)*e+.15*g.axis,(21,1)))
obs=classified_observation(synthetic,g,reg['source_known']);ref=EstimatedCurve(constants,g);ref.command(float(z['S']))
roles=EstimatedObserver(ref,reg['source_known']).update(obs,float(z['S']),.025,g)
assert obs['forces'][1].sum()==0 and obs['forces'][2].sum()==12 and roles['estimated_material_role'][5]==3
receiver_check=dict(center_inside_estimated_receiver_envelope=True,axial_normal_contact_receiver_force_N=float(obs['forces'][1].sum()),unknown_force_N=float(obs['forces'][2].sum()),material5_estimated_role=int(roles['estimated_material_role'][5]),passed=True)
before=ref.geometry.as_dict();original_command=ref.command;valid,_=original_command(float(z['S']))
def forced_private_rejection(S):
    if ref.geometry.as_dict()!=before:raise ValueError('forced saved-state proposal rejection, no dynamics')
    return original_command(S)
ref.command=forced_private_rejection
candidate=Geometry(g.entry+np.array([-.001,0.,-.001]),g.axis,g.radius-.0001)
est=SimpleNamespace(geometry=candidate,last=dict(reliable=True,latest_evidence_t=19.,t=19.,confidence=.8))
updater=GlobalUpdater(reg['global_update'],True);updater.update(ref,est,float(z['S']),20.)
history=updater.history[-1];restored,_=original_command(float(z['S']))
assert not history['written'] and len(history['proposal_failures'])==7 and history['raw_proposed_rad'] is None and ref.geometry.as_dict()==before and np.array_equal(restored,valid)
out=dict(scope='Saved integration state plus targeted synthetic contact/proposal failure checks; no mj_step or new dynamics',saved_initial_classification=rows,receiver_unknown_separation=receiver_check,all7_proposal_exceptions=dict(passed=True,restored_original_geometry=True,restored_original_command_exact=True,history=history))
(BASE/'R2_CORRECTION_CHECKS.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(receiver_check=receiver_check,all7_rejections_restore=True),indent=2))
