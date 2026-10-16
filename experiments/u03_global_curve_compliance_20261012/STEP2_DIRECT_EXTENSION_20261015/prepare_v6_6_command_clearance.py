"""Keep geometric constraints and also check the last sent local offset."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF';B=P/'CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE';B.mkdir(exist_ok=False)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
r=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));r.update(version='DIRECT_EXTENSION_V6_6_COMMAND_AWARE_CLEARANCE',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,new_dynamics_starts=0,prior_dynamics_starts_preserved=31,prior_legacy_postwrap_starts_preserved=8,prior_version='V6.5 source given drift fails27.075s; head-only contact, probeclock2.4s, no reliable fit/G/capture',revision_reason='Geometric-only clearance checks ignore local preload/probe/force offsets. Actual total-command filter therefore shortens otherwise accepted C candidates up to32fold. Retain geometric checks and also constrain C plus previous sent local offset.')
r['initial_definition']=r['initial_definition'].replace('V6.5 preserves','V6.6 preserves')
r['focused_corrections']=dict(baseline='V6.5 complete unchanged force policy and physical chain',runtime_change='DirectController sets previous total command minus current geometric C as a fixed local offset. RoughPlanner retains all geometric constraints and adds identical clearance and node-motion constraints for geometric C plus that offset.',offset_input='Only the previous sent native total command and existing geometric-command state; no actual fore FK or target truth enters G.',limitation='Previous local offset is held fixed during the private solve; new local changes still pass the unchanged final common filter. Saved-state command feasibility does not establish physical source retention or probing.')
r['command_aware_clearance']=dict(enabled=True,geometric_constraints_retained=True,previous_local_offset='previous_sent_total_native - optimization_start_geometric_native',command_constraints='Original native/source-original/rough-target/self/measured-plane and2mm C-node motion constraints, also applied to geometric+offset',current_local_change='Still protected by unchanged total command filter',no_new_limits_or_gains=True)
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(B/'register_direct.py').write_text('"""Prepared by ../prepare_v6_6_command_clearance.py; never reset a frozen case."""\n',encoding='utf-8')
f=B/'direct_controller.py';s=f.read_text(encoding='utf-8');a="        if self.source_ready and self.post is None:\n            self.theta=self.updater.update(";b="        if self.source_ready and self.post is None:\n            self.planner.command_local_offset=self.raw-self.planner.q\n            self.theta=self.updater.update(";assert s.count(a)==1;s=s.replace(a,b);f.write_text(s,encoding='utf-8')
f=B/'rough_planner.py';s=f.read_text(encoding='utf-8');a='self.coordinates=CurvatureCoordinates();self.q=self.initial_q.copy();self.snapshot=None;';b='self.coordinates=CurvatureCoordinates();self.q=self.initial_q.copy();self.command_local_offset=np.zeros(20);self.snapshot=None;';assert s.count(a)==1;s=s.replace(a,b)
a='self.last_geometric_targets=geometric_targets.copy();previous=self.q.copy();previous_points=self.fk(previous).copy();active=np.arange(n)';b='self.last_geometric_targets=geometric_targets.copy();previous=self.q.copy();offset=self.command_local_offset.copy();previous_command_points=self.fk(previous+offset).copy();previous_points=self.fk(previous).copy();active=np.arange(n)';assert s.count(a)==1;s=s.replace(a,b)
start=s.index('            gaps=[];limits=[];gradients=[];legacy_gaps,legacy_points,legacy_normals=self.legacy_source_contacts(n)');end=s.index('            cache.update(source_axis_balance_moment_Nm=',start);oldblock=s[start:end]
clearance=oldblock[oldblock.index('            for a,b,kind,i in pairs:'):oldblock.index('            motion=np.empty(0);')]
clearance='\n'.join('    '+line if line else line for line in clearance.rstrip('\n').split('\n'))+'\n'
newblock='''            # Keep every original geometric constraint. Also check the total
            # command represented by C plus the previous sent local offset.
            gaps=[];limits=[];gradients=[];constraint_labels=[];motions=[];motion_jacobians=[];geometric_gap_count=0
            for chart in ['geometric','command']:
                if chart=='command' and not np.any(offset):continue
                constraint_points=self.fk(q if chart=='geometric' else q+offset)
                jpa=np.zeros((3,self.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(self.m,self.d,jpa,jra,self.snapshot['anchorP'],self.head+self.anchor);jpa=jpa[:,self.va[:n]];jra=jra[:,self.va[:n]]
                legacy_gaps,legacy_points,legacy_normals=self.legacy_source_contacts(n)
'''+clearance+'''                constraint_labels.extend([(chart,)+pair for pair in pairs]+[(chart,'measured_plane',ct['material']) for ct in self.contact_planes if ct['material']<n])
                if chart=='geometric':geometric_gap_count=len(gaps)
                if position_cap is not None:
                    reference=previous_points if chart=='geometric' else previous_command_points
                    displacement=constraint_points[:n]-reference[:n];Jpoints=np.array([self.point_J(constraint_points[i],self.head+i,jpa,jra,n)[0] for i in range(n)])
                    motions.append(1-np.sum(displacement**2,axis=1)/position_cap**2);motion_jacobians.append(-2*np.einsum('ij,ijk->ik',displacement,Jpoints)/position_cap**2)
            motion=np.concatenate(motions) if motions else np.empty(0);motion_J=np.vstack(motion_jacobians) if motions else np.empty((0,n))
'''
s=s[:start]+newblock+s[end:];a='cache.update(source_axis_balance_moment_Nm=';b='cache.update(constraint_labels=constraint_labels,geometric_gap_count=geometric_gap_count,source_axis_balance_moment_Nm=';assert s.count(a)==1;s=s.replace(a,b)
a="            most_active_clearance_pair=str(pairs[int(np.argmax(cache['limits'][:len(pairs)]-cache['gaps'][:len(pairs)]))]))";b="            geometric_clearance_violation_m=float(np.max(cache['limits'][:cache['geometric_gap_count']]-cache['gaps'][:cache['geometric_gap_count']])),command_clearance_violation_m=float(np.max(cache['limits'][cache['geometric_gap_count']:]-cache['gaps'][cache['geometric_gap_count']:])) if len(cache['gaps'])>cache['geometric_gap_count'] else None,previous_local_offset_peak_rad=float(abs(offset).max()),most_active_clearance_pair=str(cache['constraint_labels'][int(np.argmax(cache['limits']-cache['gaps']))]))";assert s.count(a)==1;s=s.replace(a,b);f.write_text(s,encoding='utf-8')
print(str(B))
