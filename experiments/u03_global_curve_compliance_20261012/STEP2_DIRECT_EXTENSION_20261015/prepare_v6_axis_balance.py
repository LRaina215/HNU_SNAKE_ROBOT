"""Preserve original grip and reduce the unactuated source-axis load in geometric C."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS';B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';B.mkdir(exist_ok=False)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
r=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));r.update(version='DIRECT_EXTENSION_V6_SOURCE_AXIS_BALANCE',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,new_dynamics_starts=0,prior_dynamics_starts_preserved=25,prior_legacy_postwrap_starts_preserved=8,prior_version='V3.3 original given-grip fore/native chain; V4..V5.1 source torque prototypes and all failures preserved',revision_reason='Original soft elliptic contacts are not rigid source anchors. At the last saved pose static inverse dynamics requires0.373Nm unactuated source-axis torque. Reduce the net source-axis gravity/observed non-source moment by fore C shape while preserving original rear position grip. No given/physics change.')
r['kinematic_solver']['source_axis_balance']=dict(enabled=True,moment_scale_Nm=.02,activation_alpha=.1,target_net_moment_Nm=0.,source_axis='Known source +X at registered center_yz',measured_non_source_loads='Existing unlabeled world contact forces, classified only against known source; source friction loads are not credited',mass_scope='Only21 robot bodies,2.856kg',scope='Geometric C only during alpha/beta; immutable independent G, original native integral and original local front/probe/source-grip servo chain retained')
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(B/'register_direct.py').write_text('"""Registration prepared by ../prepare_v6_axis_balance.py; never reset a frozen case."""\n',encoding='utf-8')
f=B/'rough_planner.py';s=f.read_text(encoding='utf-8').replace('from native_mapping import NativeMapping,LINK','from native_mapping import NativeMapping,LINK\nfrom contact_sensor import known_source')
s=s.replace('self.initial_q=initial_qpos[self.qa].copy()','self.initial_q=initial_qpos[self.qa].copy();self.observed_external_source_X_moment_Nm=0.')
s=s.replace("self.d.qpos[:]=self.snapshot['qpos'];self.position();self.contact_planes=[]", "self.d.qpos[:]=self.snapshot['qpos'];self.position();self.contact_planes=[]\n        origin=np.r_[0.,self.p['source_known']['center_yz_m']];self.observed_external_source_X_moment_Nm=sum(float(np.cross(ct['point']-.5*ct['penetration_m']*ct['normal']-origin,ct['world_force_N'])[0]) for ct in contacts if not known_source(ct,self.p['source_known']))")
needle="            gaps=[];limits=[];gradients=[];legacy_gaps,legacy_points,legacy_normals=self.legacy_source_contacts(n)"
insert="""            balance=self.p.get('source_axis_balance',{});balance_moment=0.;balance_gradient=np.zeros(n);balance_activation=0.
            if balance.get('enabled',False):
                origin=np.r_[0.,self.p['source_known']['center_yz_m']]
                for i in range(21):
                    mass=self.m.body_mass[self.head+i];force=mass*self.m.opt.gravity;point=self.d.xipos[self.head+i];balance_moment+=np.cross(point-origin,force)[0]
                    if i<n:
                        Jp,_=self.point_J(point,self.head+i,jpa,jra,n);balance_gradient+=force[2]*Jp[1]-force[1]*Jp[2]
                balance_moment+=self.observed_external_source_X_moment_Nm;balance_activation=NativeMapping.smooth(alpha/balance['activation_alpha'])
                factor=balance_activation/balance['moment_scale_Nm'];residual.append(np.array([factor*(balance_moment-balance['target_net_moment_Nm'])]));jac.append((factor*balance_gradient)[None,:])
"""
if s.count(needle)!=1:raise RuntimeError('Unexpected objective position')
s=s.replace(needle,insert+needle)
s=s.replace("cache.update(q=qfront.copy(),r=np.concatenate(residual)","cache.update(source_axis_balance_moment_Nm=balance_moment,source_axis_balance_activation=balance_activation,q=qfront.copy(),r=np.concatenate(residual)")
s=s.replace("head_error_m=float(np.linalg.norm(cache['points'][0]-target)),", "head_error_m=float(np.linalg.norm(cache['points'][0]-target)),source_axis_balance_moment_Nm=float(cache['source_axis_balance_moment_Nm']),source_axis_balance_activation=float(cache['source_axis_balance_activation']),")
f.write_text(s,encoding='utf-8');print(str(B))
