# V6.6.1 correct empty added-contact diagnostics

Independent review rejected V6.6 before any integration: at the saved actual15.2s pose, measured contacts are onlyM13/14/16/18/20, so none is in the released first8 window. The newly logged command_incremental_clearance_violation_m attempted np.max on the empty added-plane array and raised ValueError. The rejected frozen branch and review remain intact. Waiter81953 has exited; no V6.6 physical start occurred.

V6.6.1 changes only the final rough_planner.solve diagnostic: keep the original maximum for a nonempty added-plane array; return0 when empty, and log command_incremental_plane_count. The optimization, all residuals/Jacobians/constraints/bounds/acceptance/restoration, returned commands and every other runtime module are byte-identical to frozen V6.6. No physical or safety change. A0 violation with count0 means no added plane is eligible, not that contacts were acquired.

EMPTY_CONTACT_REGRESSION.json uses the same actual15.2s physical pose/contact set as the independent rejection, with a saved-command private chart and identical saved controller-generated targets for both versions. Old V6.6 reproduces the same empty-array ValueError. V6.6.1 returns normally, accepted tracking with plane_count0 and added-plane diagnostic0. A22.3s nonempty comparison has4 eligible contacts and exactly0rad difference between old/new returned native commands. Actual snapshot qpos remains unchanged; simulation_steps0. These are required regressions, not a physical rollout or proof of new contact/source retention. CONTROLLER_ENTRY_AUDIT retains the original t0/native/source10 checks. Unchanged new-plane gradient and three bounded clearance witnesses are in the prior branch and remain limited static evidence.

Request renewed independent review for one original physical-t0 T_UPDATE diagnostic; current flagsfalse/starts0, cumulative39 actual starts unchanged. Full review of the V6.6 constraints and all actual action outcomes remain necessary. No final4 or completion claim.

The retained V6.6 design follows; references to its0 starts remain historical but its runtime mathematics are unchanged:

# V6.6 make private C planning aware of the previous sent local offset

V6.5 reaches real receiver contact but stops27.075s at the original12mm source guard; only material0 contacts and probe clock2.4s. Its new requested relief activates but does not establish reduced pressure: actual receiver downward load peaks4.509N. The full actual record is ../V6_5_ACTUAL_DIAGNOSTIC.md.

## Observed inconsistency

RoughPlanner previously checks clearance and measured planes using geometric C alone, then DirectController adds preload/probe/local force fields. The unchanged common filter checks their total and can shorten it much further. Actual22.3s: initial rate fraction0.330269, extra measured-plane compression0.777595mm, final factor0.020642 and full geometry step0.087476mm. The algebraic saved-command reconstruction agrees with the recorded final step. This is a verified mismatch between two command charts; its contribution to physical source drift is not isolated.

## Exact final runtime change

Before the private C solve, after its existing posterior actual-joint boundary assignment, DirectController sets planner.command_local_offset = previous_sent_total_native - optimization_start_geometric_native. This uses existing command history only. Consequently previous_geometric+offset equals the previous sent total command, including the compensating posterior preload. During that private solve the offset is fixed.

RoughPlanner retains EVERY original geometric objective, native/source-original/rough-target/self/measured-plane constraint, original bounds and acceptance/restoration rule. It additionally evaluates measured contact planes at geometric_native+offset, with the SAME limit as the final common filter: min(-0.8mm, previous_total_command_plane_gap-0.1mm). It also retains the original2mm geometric node-step constraints and adds2mm total-command node-step constraints. Analytic Jacobians are evaluated in the respective command posture with the same fixed measured bearing. New constraint residual penalties are zero when those constraints are satisfied. Original clearances are never relaxed.

The total-command plane rule is incremental, not an absolute collision demand. Existing virtual penetration from servo preload is the preserved baseline, exactly as in the original common filter. Two rejected static prototypes are preserved: absolute collision checks made the loaded initial command infeasible; extra original-collision command Jacobians were unreliable at a saved virtual overlap, so that addition was removed. The final extra constraints use only measured planes with a finite-difference-verified Jacobian. Original geometric source-original collision and final total-command source-original protection remain intact.

G construction, geometric head/body/tangent goals, gravity balance objective, estimator/G updates, physical probes, V6.5 local force policy and all local fields remain separate and unchanged. No actual fore-body FK enters G. Native transmitted-history recurrence and actual-state input boundary are unchanged. New local changes after the private solve are still checked by the same final common filter. No physics, initial, actuator, source12mm/12degree, rate, curvature, joint, contact/floor/self or capture threshold change. No source feed/roll before measured capture.

## Limited checks

Entry audit passes original10 checks without integration. COMMAND_CLEARANCE_AUDIT compares one bounded old/new solve at each saved2.1/22.3/25s actual state with identical saved controller-generated G/C goals and fixed previous local offset. New candidates are accepted. Common factors old/new:0.06655/0.38768,0.04820/0.38643,0.03125/0.38532; new sent full geometry steps0.615/1.117/1.205mm, measured-plane/source excess0. The22.3s/25s new solves remain tracking accepted. These are isolated private witnesses, not replayed dynamics or full-path feasibility. New measured-plane Jacobian FD error4.51e-8 in solver-scaled coordinates at22.3s; actual qpos is not modified. No integration performed for this version.

Request one original physical-t0 T_UPDATE development diagnostic after exact independent review. Preserve all39 prior actual starts. Current flagsfalse and starts0. No final-four comparison or Step2 completion acceptance requested; actual source retention/contact/probes/identification/new bearing/whole transfer and50s continuation remain unvalidated.
