"""Restore actual Step1 t0/coordination; add only Step2 prior/evidence/G inputs."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;F=P.parent/'STEP1_20261014/R1_READY_FROZEN';old=P/'CONTROLLER_V9_1_PRIOR_DOMAIN_BEARING';B=P/'CONTROLLER_V11_STEP1_ORIGINAL_ROLLING';B.mkdir(exist_ok=False)
for name in ['physics.py','geometry.py','native_mapping.py','source_mode.py','contact_sensor.py','cylinder_estimator.py','estimated_observer.py','estimated_curve.py','evaluation.py','visual_overlay.py','native_updater.py','action_stop_evaluator.py']:
    src=(P/'CONTROLLER_V7_FINAL_CONTRACT_R1'/name) if name=='estimated_curve.py' else old/name
    shutil.copy2(src,B/name)
shutil.copy2(P.parent/'STEP2_FROM_SOURCE_20261015/from_source_curve.py',B/'from_source_curve.py')
shutil.copy2(F.parent/'N_SHAPE/initial.npz',B/'initial.npz');shutil.copy2(P/'STEP1_ORIGINAL_SOURCE_CONSTANTS.json',B/'STEP1_SOURCE_CONSTANTS.json')
shutil.copy2(P/'native_mapping_vectorized.py',B/'native_mapping_vectorized.py');shutil.copy2(P/'NATIVE_VECTOR_EQUIVALENCE.json',B/'NATIVE_VECTOR_EQUIVALENCE.json')
f=B/'estimated_curve.py';s=f.read_text();s=s.replace('from native_mapping import NativeMapping,LINK','from native_mapping_vectorized import FastNativeMapping as NativeMapping,LINK');s=s.replace('from source_mode import SourceMode,reconstruct','from source_mode import SourceMode,reconstruct\nfrom contact_sensor import receiver_axial_lower')
s=s.replace('(axial>-.05)','(axial>receiver_axial_lower(self.geometry,self.source_known))')
f.write_text(s,encoding='utf-8')
f=B/'from_source_curve.py';s=f.read_text();start=s.index('    def configure_initial(');end=s.index('    def command(self,S):',start);s=s[:start]+s[end:]
start=s.index('        if self.initial_bias is not None and S<.75:');end=s.index('        v=np.clip(mid/nodes[8]',start);s=s[:start]+s[end:]
s=s.replace(",initial_preload_fraction=float(1-self.smooth(S/.75))",',initial_preload_fraction=0.')
s+='''

    def nominal_command(self,S):
        delta=self.delta;gamma=self.source_gamma;probe=self.probe.copy();relief=self.front_relief.copy()
        self.delta=0.;self.source_gamma=0.;self.probe[:]=0.;self.front_relief[:]=0.
        try:return self.command(S)
        finally:self.delta=delta;self.source_gamma=gamma;self.probe=probe;self.front_relief=relief
'''
f.write_text(s,encoding='utf-8')
f=B/'physics.py';s=f.read_text();s+='\nINITIAL=BASE/"initial.npz"\n';f.write_text(s)
r=json.loads((old/'REGISTERED.json').read_text());constants=json.loads((B/'STEP1_SOURCE_CONSTANTS.json').read_text())['constants']
r.update(version='STEP2_V11_STEP1_ORIGINAL_ROLLING',requirements='STEP2_ROLLING_SCOPE_CLARIFIED_20261016',autonomy='STEP2_AUTONOMOUS_NO_PROCESS_HEARTBEAT_20261016',execution_backend='Original A16 continuous release/F/M/material gauge/common roll from original Step1 physical t0; Step2 prior/estimation/G writes and contact-based B1 C',source_constants=constants,duration_s=235.,initial_file='initial.npz',initial_source=str(F.parent/'N_SHAPE/initial.npz'),dynamics_authorized=True,current_code_self_checked=False,all_previous_actual_starts=48,new_dynamics_starts=0,per_version_supervisor_gate_required=False,process_heartbeat=False,revision_reason='User corrected rolling scope and original initialization: normal advance/rolling after the placed t0 is allowed. Withdraw no-feed-before-capture, changed initial, fixed-phase6 and replacement execution patches. Preserve all48 prior starts as historical failures/partial evidence. Directly restore Step1 complete release coordination, with rough prior and contact-driven G/C instead of target truth.')
r['settle_s']=.5;r['nominal_frequency_hz']=.02;r['command_interval_s']=.025
r['initial_definition']='Exact original Step1 N_SHAPE physical t0 qpos/qvel/ctrl; all21 on source, no receiving pre-wrap; fresh data loaded exactly as Step1 runner, no intermediate checkpoint or joint replay'
r['normal_progression']='S=feed_per_cycle*0.02*max(0,t-0.5), F/M/common psi and grow(S/6) all original; normal precontact/prebearing advance is allowed, no locks'
r['source_constants_provenance']='Known-source helix and material calibration from successful Step1 t0; no receiving pose/radius/trajectory in these constants'
r['local_C']='Original B1 receiving radial deformation +/-2mm and late-source0.4/m mode; observation uses permitted pose/unlabeled contacts and applied G frame. Bounded two-direction probes added after real front contact.'
r['capture_motion_evidence'].update(role='Observed handoff evidence; does not lock normal precontact progression',minimum_loaded_s=2.,minimum_receiving_vertical_fraction=.4,front_motion_hinges=8)
for key in ['direct_extension','native_shape','source_balance','handoff','local_stance']:
    if key in r:r['historical_'+key]=r.pop(key)
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
unused=P/'CONTROLLER_V10_1_MOVING_LOCAL_WINDOW';(unused/'NOT_SELECTED_ROUTE.md').write_text('Superseded before any integration by user STEP2_ROLLING_SCOPE_CLARIFIED_20261016. Current changed initial/no-precontact-feed constraints were not user requirements. Preserve preparation as0starts, do not launch.\n',encoding='utf-8')
print(str(B))
