"""Freeze one mechanically motivated candidate and disclose witness limitations."""
from pathlib import Path
import json,difflib,datetime
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';old=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS'
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'))
r['initial_definition']='Exact original limited-reviewed given source pose, all21 source-side, no receiving contact/prewrap. This is not exact static equilibrium. V6 preserves this initial and original V3.3 source position grip; current code requires independent review.'
r['focused_corrections']=dict(baseline='V3.3 given source position grip; its gravity sign and known source seam classification retained',change='Add one soft fore C moment residual and measured unlabeled non-source force-moment input; source torque controllers V4..V5.1 not included',physics_basis='Original elliptic soft contacts cannot be treated as rigid source anchors; exact-pose inverse audit exhibits an unactuated source-axis residual',limits='Balance is a soft objective; isolated witness tangent errors37..66deg and final moment0.076Nm remain. No path or physical feasibility claim.')
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(B/'CONTROLLER_DESIGN.md').write_text('''# V6 source-axis balance: one development diagnostic only

This candidate restores the original V3.3 given posterior position grip. V3.3, V4.1, V5 and V5.1 all failed the original source drift guard; V5.1 stopped at13.9s before any receiving contact. No Step2 action success has been established. V4..V5.1 rigid-source force/inverse controllers remain archived, and are not imported here.

The original MuJoCo3.2.7 elliptic contact model is soft. At four saved V5.1 poses, exact delivered requested joint torque still differs from rigid-source body acceleration predictions by0.043..0.388m/s2. At the13.9s exact pose with velocity/acceleration zero, inverse dynamics requires0.372853Nm about known sourceX; source joint actuators have zero virtual work on rigid global roll with all joints fixed. This rules out exact static equilibrium at those poses, not all reachable poses or compliance with a finite12mm tolerance. See ../CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION/T_UPDATE/SOFT_CONTACT_COMPATIBILITY_AUDIT.json and the [official3.2.7 soft-contact equations](https://mujoco.readthedocs.io/en/3.2.7/computation/#parameters).

## Exact change relative to V3.3

Only rough_planner.py runtime behavior changes. C adds residual a(alpha)*(Mx_gravity+Mx_measured_non_source)/0.02Nm. The activation is the existing smoothstep(alpha/0.1), zero at entry. Gravity moment sums21 robot-body COM loads, total2.856kg, about the known source+X line at YZ=(0,1). The optimizer changes only currently released fore hinges, with the existing fixed-bearing analytical Jacobian. Source body masses are included in the value but their geometry is fixed during each solve. The gradient was checked at three poses, maximum absolute error1.382e-8Nm/rad.

Existing unlabeled non-source contact world forces contribute their observed source-axis moment. The sensor environment-surface point is shifted by half penetration to its original contact midpoint. Known-source forces are not credited as supporting moment. No receiver names, true cylinder geometry, true mesh, or saved paths enter this term. Unmeasured contact torques and varying contact force within an optimization step are not predicted; this is a geometric load-reduction heuristic, not a full dynamics balance solution.

The independent G builder is untouched. It still uses immutable given geometry and rough hypotheses, with fixed64.5mm material coordinates and native curvature integrals. Source q/control retain given position grip as V3.3. Original fore gravity/local relief/probe layers, contact gating, fit/G update/capture criteria, alpha/beta history, postcapture Step1 mapping and no feed/roll before capture are untouched. Original initial file, MuJoCo flags0, solver/contacts/friction/mass/actuators and all actual12mm/12degree source, penetration, self/floor/joint/torque safeguards are untouched.

## Limited evidence and unresolved risks

CONTROLLER_ENTRY_AUDIT.json passes: t0/hold commands unchanged, no actual pose writes, source feed/roll/probe zero and native integral correct. BALANCE_GRADIENT_ENTRY_AUDIT.json passes the added gradient, entry activation and one explicit unlabeled-force input check. All these have zero integration steps.

BALANCE_WITNESSES.json uses three isolated alpha0.3/0.6/0.888 poses, seeded from the old coarse-prior no-integration prototype only. Runtime does not load these seeds. Net moment decreases from-0.248/-0.419/-0.479Nm to-0.0169/-0.0106/-0.0762Nm while source nodes remain fixed. Head position error is0.206/0.155/3.308mm; head tangent error is37.0/65.6/60.3degrees against the actual solver guide chord. The last local solve is not successful. These are limited load-reduction witnesses, not a feasible full path, valid touchdown orientation, adequate source dynamics, or receiver contact proof. No tolerance has been relaxed to classify them as success.

No new initial search or full offline IK sweep is requested. A single original physical-t0 T_UPDATE diagnostic, if independently approved, will measure whether the added objective reduces actual source drift and allows real contact. Failure and all actual states/video/contact/probe evidence must be retained. No final four-case comparison is authorized by this packet; it follows only after actual complete-action evidence and a separate frozen comparison.

## Runtime guard

direct_runner.py refuses integration without exact-version independent approval, original reviewed initial binding and byte-identical frozen Python/registration/model/assets (no SHA). Three approval booleans are the only registration fields permitted to change after freezing. This script never grants approval.
''',encoding='utf-8')
s=(B/'prepare_review_snapshot.py').read_text(encoding='utf-8')
start=s.index("    evidence=[");end=s.index("    limitations=",start)
s=s[:start]+"    evidence=['CONTROLLER_DESIGN.md','CONTROLLER_ENTRY_AUDIT.json','BALANCE_GRADIENT_ENTRY_AUDIT.json','BALANCE_WITNESSES.json','../V5_1_ACTUAL_DIAGNOSTIC.md','../CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION/T_UPDATE/SOFT_CONTACT_COMPATIBILITY_AUDIT.json'],\n"+s[end:]
s=s.replace("'Wrap prototypes are mechanical diagnostics, not sensor/fit/capture success'","'Isolated balance witnesses retain37..66degree head tangent errors; not a valid full path or touchdown proof'")
(B/'prepare_review_snapshot.py').write_text(s,encoding='utf-8')
diff=[]
for name in ['rough_planner.py','register_direct.py','prepare_review_snapshot.py']:
    diff.extend(difflib.unified_diff((old/name).read_text(encoding='utf-8').splitlines(True),(B/name).read_text(encoding='utf-8').splitlines(True),fromfile='V3.3/'+name,tofile='V6/'+name))
(B/'V6_CHANGES.diff').write_text(''.join(diff),encoding='utf-8')
(P/'V5_1_ACTUAL_DIAGNOSTIC.md').write_text('''# V5.1 actual failure and soft-contact compatibility

The original physical-t0 T_UPDATE stopped13.900s at source_boundary_drift_from_given. Release occurred2.100s; alpha0.472, beta/feed/roll0. No target contact, reliable geometry, G write, capture, transfer or post-transfer roll. Exact saved integration-state source body-origin drift is12.042325mm from given and11.016368mm from the stable reference; the10Hz trace stop metric is12.040711mm from given. The difference is the final integration-state measurement, not a pass.

The nominal grip projection prevents the rapid preload relaxation observed in V5: allocated retained normal load87.584N, measured88.508N at stop. Posterior control changes2.878degrees and posterior joint changes2.855degrees at the last sample, while mean source polar error is-5.618degrees. This is a different failure trajectory; the whole action still fails. Raw result/trace/contact evidence,140-frame continuous video, visually checked contact sheet, exact stop image and21-rod exact-motion CSV are retained in the case folder. The action evaluator correctly reports source guard failure.

Read-only reconstruction of the original flag0 source-only model delivers the requested joint torques to numerical precision, but actual soft-contact source-body acceleration differs from the rigid-source QP prediction by0.0434/0.2388/0.2166/0.3877m/s2 at2.1/5/10/13.9s. This establishes a modeling mismatch, without isolating its share in the observed failure. Zero-velocity/acceleration inverse dynamics at those exact poses requires unactuated rigid source-axis moments0.0978/0.1250/0.2786/0.3729Nm; joint-actuator virtual work on global rigid roll is zero and inverse friction loads are zero there. It does not prove every reachable shape, the original placement, or finite12mm stationary tolerance infeasible.

The [official MuJoCo3.2.7 computation model](https://mujoco.readthedocs.io/en/3.2.7/computation/#parameters) uses positive constraint regularization and zero position residual in friction dimensions. A rigid zero-acceleration source constraint is therefore not the original soft-contact plant. Original physics is unchanged. The next concrete candidate reduces required source-axis load through free fore C geometry while retaining original posterior grip; it remains unvalidated.

Cumulative actual starts33 (20 historical source,8 old postwrap,5 direct). All completed action successes0. No active actual process or gate waiter at preparation time. No final4 yet.
''',encoding='utf-8')
print('V6 design, exact diff, and V5.1 report saved; freeze with branch prepare_review_snapshot.py next')
