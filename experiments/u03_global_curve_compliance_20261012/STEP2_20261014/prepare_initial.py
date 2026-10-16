"""Construct a given complete initial, source constants and read-only checks."""
from physics import *
import sys,pickle,json,shutil,xml.etree.ElementTree as ET
old=BASE.parent/'STEP1_20261014';sys.path.append(str(old))
with (old/'N_SHAPE/controller_checkpoint.pkl').open('rb') as fp: legacy=pickle.load(fp)['reference']
keys=['radius','pitch','phi0','x0','beta','a','feed_per_cycle','feed_early','feed_late','tail_u0','anchor_u0','supply_anchor_normal','gauge_angle']
constants={key:getattr(legacy,key).tolist() if isinstance(getattr(legacy,key),np.ndarray) else float(getattr(legacy,key)) for key in keys}
constants['source_endpoint_x_m']=.95
(BASE/'source_constants.json').write_text(json.dumps(constants,indent=2))
z=np.load(old/'N_SHAPE/trace.npz');idx=int(np.argmin(abs(z['t']-90.)))
initial=BASE/'GIVEN_INITIAL';initial.mkdir(exist_ok=True);shutil.copy2(MODEL,initial/'model.xml')
for mesh in ET.parse(MODEL).findall('./asset/mesh'):
    if mesh.get('file'):
        dest=initial/mesh.get('file');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MODEL.parent/mesh.get('file'),dest)
m=mujoco.MjModel.from_xml_path(str(initial/'model.xml'));d=mujoco.MjData(m)
d.qpos[:]=z['qpos'][idx];d.qvel[:]=z['qvel'][idx];d.ctrl[:]=z['ctrl'][idx];d.time=0.;mujoco.mj_forward(m,d)
state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez(initial/'integration_state.npz',state=state,qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,S=float(z['S'][idx]),receiving_delta_m=float(z['compliance_delta_m'][idx]))
from contact_sensor import read_sensor,known_source
from geometry import Geometry
from estimated_curve import EstimatedCurve
registered=json.loads((BASE/'REGISTERED.json').read_text());sensor=read_sensor(m,d);checks=[]
legacy.delta=0.;legacy.source_gamma=0.
nominal=Geometry([1.1,.2,1.],[0,1,0],registered['source_known']['radius_m'])
ref=EstimatedCurve(constants,nominal)
for S in [float(z['S'][idx]),18.,23.,35.]:
    q,info=ref.command(S);oq,oi=legacy.command(S)
    checks.append(dict(S=S,raw_max_difference_rad=float(np.max(abs(q-oq))),G_material_max_difference_m=float(np.max(np.linalg.norm(info['global_reference_points']-oi['global_reference_points'],axis=1)))))
reachable=[]
for name,prior in registered['conditions'].items():
    g=Geometry.from_dict(prior);eligible=[]
    for ct in sensor['contacts']:
        if known_source(ct,registered['source_known']):continue
        axial,_,r=g.coordinates(ct['point'])
        if -.10<axial<.90 and abs(r-g.radius)<.03 and abs(ct['normal']@g.axis)<.45:eligible.append(ct)
    q,_=EstimatedCurve(constants,g).command(float(z['S'][idx]))
    reachable.append(dict(condition=name,eligible_initial_contact_count=len(eligible),material_ids=sorted(set(x['material'] for x in eligible)),
        prior_radial_surface_residual_range_m=[float(min(abs(g.coordinates(x['point'])[2]-g.radius) for x in eligible)),float(max(abs(g.coordinates(x['point'])[2]-g.radius) for x in eligible))] if eligible else None,
        raw_prior_to_initial_ctrl_max_deg=float(np.degrees(np.max(abs(q-d.ctrl))))))
out=dict(source_trace='STEP1_20261014/N_SHAPE/trace.npz',saved_sample_t=float(z['t'][idx]),given_initial_reference_S=float(z['S'][idx]),initial_ancillary_state='fresh MjData; qpos/qvel/ctrl copied; forward once; complete integration state saved including warmstart; both paired variants identical',historical_continuation_claim=False,
    original_model_equal=(initial/'model.xml').read_bytes()==MODEL.read_bytes(),nominal_mapping_equivalence=checks,prior_reachability=reachable,
    scope='saved states and private kinematics only; no new physics steps; exact geometry used only to validate frozen mapping equivalence')
(BASE/'PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
assert all(x['eligible_initial_contact_count']>=2 for x in reachable)
assert all(x['raw_max_difference_rad']<1e-8 for x in checks)
