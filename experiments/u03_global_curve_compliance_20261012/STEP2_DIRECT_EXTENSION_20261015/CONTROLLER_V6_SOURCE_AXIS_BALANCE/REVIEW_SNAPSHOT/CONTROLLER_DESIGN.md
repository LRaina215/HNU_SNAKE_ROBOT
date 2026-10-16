# V6 source-axis balance: one development diagnostic only

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
