# V6.7 receiving wrap chart begins at the prescribed early-contact entry

V6.6.1 runs physically from originalt0 without the empty-plane error, but stops25.625s at the original12mm source guard. It remains head-only contact/probeclock2.775s with no reliable geometry, G write or new bearing. Actual receiver downward load peaks5.349N. Improved command transmission is insufficient. See ../V6_6_1_ACTUAL_DIAGNOSTIC.md and preserved full actual record.

## Confirmed target/chart mismatch

First contact occurs at alpha0.624, before the nominal extension endpoint. The prescribed G goal there is phase-129.35544deg and axial-54.25690mm relative the rough receiver. Wrap construction nevertheless starts its internal receiving family at fixed-165deg/axial0. Separate radius interpolation from pipe radius toward receiving radii is followed by the existing outer transition from the event curve, so the nominal family is interpolated twice. Saved actual C head targets shrink from the event radius83.328mm to78.017mm, and at maximum downward force the head task error points5.656mm inward along the measured contact normal. This motivates a geometry correction; it does not establish sole causality for drift.

## Exact runtime changes: RoughPlanner only

start_wrap already stores the prescribed independent G at the contact event and the previous commanded C separately. It now additionally stores phase_origin=atan2(prescribed_G_goal radial coordinates)-current_theta and axial_origin=prescribed_G_goal axial coordinate in the rough cylinder chart. These origins use NO commanded/actual body FK. The actual contact event selects the controller phase, as before; its measured point does not populate G.

wrap_guide uses phase_origin+theta rather than the unused nominal terminal phase. Its receiving-family axial coordinate includes axial_origin plus the original pitch advance and20mm front-only axial lead. Virtual G uses its original full rough_radius+39mm family. Model-aware native C uses the original immutable known-source node radii shifted only by rough_radius-known_source_radius. The internal pipe-radius interpolation is removed from both. The existing smooth beta/0.2 outer transition from saved prescribed G/commanded C is retained, so there is one continuous boundary transition. Existing source retention/bridge geometry,64.5mm foreground closure, coiling curvature mode and all IK criteria remain.

nominal_contact_phase includes the prescribed axial origin in polar-minus-signed2pi*axial/pitch. This keeps the estimator/updater circular-phase chart consistent with the receiving family; front-only lead subtraction and all confidence/write bounds remain. Postcapture receives this same nominal phase through the unchanged method call.

No other runtime module changes. DirectController, previous-transmitted C state, command-aware plane/node constraints and empty-set fix, common native filter, source balance cost, local force/probe fields, timing, estimator/capture/transport, source feed/roll lock and original source12mm/12degree/physics/actuator/penetration/self/floor constraints remain. Given initial identical; no prewrap or new initial search. G stays independent of actual FK.

## Limited verification

Original entry10 checks pass without integration. CONTACT_CHART_AUDIT samples five beta values (near0,0.05,0.1,0.15,0.2) in a reconstructed saved event command chart, without full-path IK. New G/C head target radii do not dip below the event radii at those samples; C radius at beta0.1 is89.100mm versus old79.504mm. New C phase advances13.032deg from the prescribed event at beta0.1, versus old retreat5.528deg. Its axial coordinate is-43.398mm versus old-16.269mm. At beta0.2 new C phase advance48deg, as prescribed. Near-zero target boundary error is0 to saved G/C starts. All G spans and optimized first14 C target spans are64.5mm to~1e-16m. The unused posterior native target suffix retains the pre-existing up-to20.65um chord error from pose blending in both versions; this is distinct from exact hardware C/native integration. The initial broader full-target span audit is preserved to expose that distinction, not hidden.

Changing initial commanded C by0.05rad leaves prescribed G exactly unchanged. Two full-family circular-phase identities at beta0.2/0.4,theta0.03 have errors<3e-16rad. These are mathematical/kinematic checks only. They do not prove physical contact retention, pressure reduction, multi-material evidence, source hold, wrap/bearing, full transfer or50s rolling. No integration has been performed for V6.7.

Request one original physical-t0 T_UPDATE diagnostic after exact independent review. All40 starts/failures retained, current starts0/all3flagsfalse. No final-four comparison or completion acceptance requested here.
