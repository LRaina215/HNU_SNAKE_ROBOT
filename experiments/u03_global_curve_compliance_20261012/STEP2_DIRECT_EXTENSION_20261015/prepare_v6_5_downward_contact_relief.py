"""Activate existing bounded local relief for low downward non-source rod loads."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V6_4_NOMINAL_PHASE_TASK_GATE';B=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF';B.mkdir(exist_ok=False)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
r=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));r.update(version='DIRECT_EXTENSION_V6_5_DOWNWARD_CONTACT_RELIEF',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,new_dynamics_starts=0,prior_dynamics_starts_preserved=30,prior_legacy_postwrap_starts_preserved=8,prior_version='V6.4 failed26.875s after true receiving contact, only material0, probeclock2.5s, no fit/capture; negative receivingFz peak3.13N and force relief identically0',revision_reason='Existing12N per-contact relief ignores several-newton downward fore load. Add pre-capture non-source net-downward per-rod relief above robotW/21 while preserving original gain/rate/cap and all other force policy.')
r['focused_corrections']=dict(baseline='V6.4 nominal task-gated phase and unchanged C/history/probe chain',runtime_change='support_policy.local_front only: group non-source fore contacts by material before capture; if net worldFz<0, use aggregate normal threshold robotW/21 and load-weighted normal Jacobian direction.',software_threshold_N=r['robot_weight_N']/21,old_cases='Known source, net nonnegative non-source load, postcapture and M8+ retain original per-contact12N relief rule.',same_gain_rate_cap='Existing0.012 curvature/N,0.06/m/s slew,0.30/m norm cap and first8 window unchanged',scope='Measured unlabeled contact loads and known source classification only; no true receiving labels/geometry or physical-state writes.')
r['downward_contact_relief']=dict(enabled=True,scope='Pre-capture M0..M7 non-source contacts with aggregate worldFz<0 per material',normal_threshold_N=r['robot_weight_N']/21,aggregation='Sum normal load per rod; sum normal-load-weighted measured normal Jacobians, then normalize once per rod',gain_per_N=.012,rate_per_m_s=.06,maximum_per_m=.30,limitations='This is a new software force policy; it does not establish actual pressure relief, physical probe delivery, wrap or source holding')
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(B/'register_direct.py').write_text('"""Prepared by ../prepare_v6_5_downward_contact_relief.py; never reset a frozen case."""\n',encoding='utf-8')
f=B/'support_policy.py';s=f.read_text(encoding='utf-8');a="""        for ct in sensor['contacts']:
            if ct['material']>=8 or ct['normal_N']<=12:continue
            jp=np.zeros((3,self.m.nv));jr=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,d,jp,jr,ct['point'],head+ct['material'])
            gradient=ct['normal']@(jp[:,columns]@mapping+jp[:,:6]@root_mapping);n=np.linalg.norm(gradient)
            if n>1e-7:desired+=.012*(ct['normal_N']-12)*gradient/n
"""
b="""        def normal_gradient(ct):
            jp=np.zeros((3,self.m.nv));jr=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,d,jp,jr,ct['point'],head+ct['material'])
            return ct['normal']@(jp[:,columns]@mapping+jp[:,:6]@root_mapping)
        non_source={}
        if self.post is None:
            for ct in sensor['contacts']:
                if ct['material']<8 and not known_source(ct,self.reg['source_known']):non_source.setdefault(ct['material'],[]).append(ct)
        downward={i:cts for i,cts in non_source.items() if sum(ct['world_force_N'][2] for ct in cts)<0.}
        for ct in sensor['contacts']:
            if ct['material']>=8 or ct['normal_N']<=12:continue
            if ct['material'] in downward and not known_source(ct,self.reg['source_known']):continue
            gradient=normal_gradient(ct);n=np.linalg.norm(gradient)
            if n>1e-7:desired+=.012*(ct['normal_N']-12)*gradient/n
        threshold=self.reg['robot_weight_N']/21.
        for cts in downward.values():
            load=sum(ct['normal_N'] for ct in cts)
            if load<=threshold:continue
            gradient=sum(ct['normal_N']*normal_gradient(ct) for ct in cts);n=np.linalg.norm(gradient)
            if n>1e-7:desired+=.012*(load-threshold)*gradient/n
"""
assert s.count(a)==1;s=s.replace(a,b);f.write_text(s,encoding='utf-8');print(str(B))
