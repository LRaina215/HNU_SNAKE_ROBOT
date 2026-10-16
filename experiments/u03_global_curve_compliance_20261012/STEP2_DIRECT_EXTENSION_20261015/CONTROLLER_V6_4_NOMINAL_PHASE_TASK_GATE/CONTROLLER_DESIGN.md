# V6.4 original nominal phase maximum under sent/actual head task gates

V6.3 restored actual C movement (25.5mm head departure and1.25mm final head task error;24 accepted restoration10Hz samples) but failed source drift11.75s before contact. StableX displacement9.12mm contributes to total given displacement12.03mm. This is not another rejected-C/alpha0 failure. Actual fore-source contacts remain loaded during a prolonged release; their role in the failure is not uniquely isolated. See ../V6_3_ACTUAL_DIAGNOSTIC.md and its saved SOURCE_AXIAL_LOAD_DIAGNOSTIC.json.

## Exact runtime change relative to frozen V6.3

One assignment in direct_controller.py changes phase_progress_fraction from last_command_transmission_factor to1. Alpha/beta therefore use their ORIGINAL registered maximum rates1/25s and1/40s when original physical joint tracking AND the V6.3 sent/actual head20mm task gate are ready. No maximum rate is increased above the original controller. With a stationary head, the task gate still bounds phase lead; it does not permit228mm divergence. Current phase fraction is logged1, and the actual common command_filter_factor remains separately logged.

The common native command filter remains fully active for raw native output, transmitted geometric C, transmitted local offset and marginal probe histories. Its curvature/servo/motion/contact checks are unchanged. The hypothesis is that this combined factor also reflects local preload transition and can unnecessarily slow the geometry phase and its alpha-driven fore unloading/balance. Removing that SECOND use of the factor may improve source retention, but this has no dynamic validation yet. It is a timing policy change, not a claim of holding or a faster successful action.

All rough_planner.py, feasibility restoration/nonlinear50um clearance acceptance, fixed material/native integral, given posterior grip, V6 source-axis moment objective, independent G builder/input boundary, force relief thresholds, probes/estimator/capture criteria and postcapture mapping are unchanged. Actual head robot state is still progress feedback only; actual whole-body geometry never fills G. Original initial/model/mass/friction/solver/actuators and source12mm/12degree, penetration/self/floor/joint/torque guards remain. Feed/rolling still require actual capture.

## Limited checks

CONTROLLER_ENTRY_AUDIT.json passes unchanged t0/hold/native/no-state-write checks. NOMINAL_PHASE_CLOCK_AUDIT.json uses the same constructed rejected-held-C stub as V6.3 with previous common fraction0.1:4.279mm head errors permit original0.001 alpha increment,63.055mm errors stop the next increment. Actual qpos is unchanged, posterior total command remains given and native integral errors<1.2e-15rad. Rough planner is byte-identical to reviewed V6.2/V6.3. These checks validate the clock/task predicate only, not physical extension or source hold.

Only one original physical-t0 T_UPDATE development diagnostic is requested after exact independent approval. All37 previous starts/failures remain; this version has0 starts and all three dynamics flagsfalse. The unchanged runner checks exact version/initial/Python/registration/model/assets by byte comparison, no SHA. No final-four comparison or completed-action acceptance is requested.
