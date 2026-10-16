# V6.5 local relief for downward non-source fore contact load

V6.4 reaches actual receiving contact and probe clock2.5s but stops26.875s at the original source12mm guard. Only head material0 contacts, receiving normal peaks4.515N and net vertical load reaches-3.135N (downward), while the existing local front-force curvature field stays exactly0. The original12N per-contact activation misses this load scale. This is an observed inactive feedback channel, not proof that pressure relief alone will fix source retention. See ../V6_4_ACTUAL_DIAGNOSTIC.md and preserved case evidence.

## Exact runtime change relative to frozen V6.4

Only support_policy.py local_front changes. Before actual capture, existing unlabeled non-source contact points on M0..M7 are grouped by robot material. If their aggregate world vertical force is negative, their NORMAL loads are summed per rod. Above mean robot rod weight W/21=1.33416N, the existing normal-force relief uses a normal-load-weighted sum of the existing fixed-bearing normal Jacobians, normalized once per rod. The existing gain0.012 curvature/N multiplies (aggregate normal load-threshold).

This is a NEW software threshold and aggregation rule. It does not change physical mass, friction, contact or actuator settings. Mean rod weight is a chosen mechanical scale, not a proven optimal contact target. Actual pressure reduction, possible contact loss and effectiveness for probing/source retention remain unvalidated.

Known-source contacts keep their original12N per-contact rule, even if a rod also has an eligible non-source contact. Non-source rods with aggregate vertical force>=0 keep the original rule, preserving positive bearing response. After capture all contacts return to the original rule. Materials M8+ stay outside the existing front relief window. Existing normal Jacobian/root elimination, source bearing selection, first8 curvature window,0.06/m/s slew and0.30/m norm cap are unchanged. No true receiver labels/axis/radius/mesh are used; only existing contact loads/normals/points and known-source classification enter.

Direct controller, nominal phase/head/joint gates, transmitted C/native/probe accounting, entire rough planner and feasibility/clearance acceptance, V6 moment objective, independent G inputs, estimator/probe/capture requirements and postcapture mapping are byte-identical to V6.4. Original initial, physical model/solver/actuators and all source12mm/12degree, penetration/self/floor/joint/torque and command motion/curvature limits are unchanged. Source feed/rolling still waits for actual capture.

## Limited verification

CONTROLLER_ENTRY_AUDIT.json passes original entry/hold/native/no-state-write checks. DOWNWARD_RELIEF_AUDIT.json uses the saved actual22.3s pose/contact observation: sensor normal3.710N/Fz-3.146N, old relief0, new first25ms relief norm0.0015/m (original slew). Private ideal following moves the measured contact point0.06963mm outward along its normal, with native offset0.00794degrees and retained M8+ native offset0. Actual qpos is unchanged. Source-only, nonnegative-load and postcapture variants have0 relief at this observation, as before. These checks establish direction, bounds and policy scope only; no integration or force-response claim.

Request one original physical-t0 T_UPDATE diagnostic after exact independent review. All38 prior starts/failures remain. Current version0starts/all three dynamics gatesfalse. Original exact runner guard compares approved version, initial, Python/registration/model/assets by bytes (no SHA). No final-four comparison or complete-action acceptance is requested here.
