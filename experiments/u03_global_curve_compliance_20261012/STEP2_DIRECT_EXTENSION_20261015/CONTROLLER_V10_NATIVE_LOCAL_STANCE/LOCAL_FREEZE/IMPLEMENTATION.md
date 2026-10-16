# STEP2_V10_NATIVE_LOCAL_STANCE development candidate

V7 actual97.1s ended without bearing: source axial slip70.55mm, G-C head gap301mm, actual fore joints followed sent command. NativeStep1Reference now directly reuses A16 source-polar C2 Bezier, receiving helix, first-forward64.5mm material chords, Bishop transport, material gauge and original curvature integrals. Source fit/gauge are one-time calibration from immutable given robot pose; original state is never substituted. Exact target is evaluation-only. RoughPlanner.solve, whole-body world target costs and command-plane/cartesian/curvature guards are not called. Original physics, limits, actuator and physical penetration/self/floor guards remain.

Two necessary adaptations are explicit. Original A16 X wait relied on source approach feed; holding source compresses the X bridge, so X is distributed across C2 Bezier controls. Original intermediate endpoint family on this different given phase demands over180deg; it is not driven. Complete initial future G uses native phase6, while C releases smoothly in native curvature coordinates from the given joint shape over25s. Source material12 and posterior retain given support commands;12to14 release may expand only after receiving load>=0.1W persists0.5s. Native source feed/material roll remain exactly0 before capture. After front geometry/loaded>=0.4W and0.5s actual aligned commanded motion>=0.25deg, inherited common S/feed/gauge roll starts, source grip fades15s, spatial phase goes6to9. These chosen numbers are not user hard limits.

Local C includes actual contact-force curvature relief, bounded two-direction probing, radial compliance up to2mm/rate0.25mm/s through the native curve, late-source measured force mode and gravity preload. A small0.1/m,4s local front trial provides loaded motion evidence. G only accepts reliable contact-fit cylinder writes at the already-registered continuity bounds; same-state native q proves it affects execution. G0/currentG/C/actual are shown together at1x. Physical coordinates for C drawing are robot kinematics anchored to current support; G never follows actual body sag or C delta. Fixed-sigma field is an exact representation of native q; the original geometric arc supports are separately logged.

Single development case only, not final four. No physical initial search, no parameter sweep, no SHA. The inherited A16 family shape phase differs from shared transport intentionally to prevent source approach rolling. No claim this adaptation already works; actual run decides. Independent final stage review remains required; per-version gate does not.


Current revision: V9.2 lost receiving bearing despite correctly pausing S0.340; actual source stance changed while fore curvature remained encoded in the original stance. Local C now joins current robot support boundary to the prescribed G helix, executes native Bishop integrals and adds a finite-rod correction around those integrals. Global G stays independent of actual body pose. On bearing, C continues on its planned physical route using original rolling/feed, with staged support gates retained.
No single-joint tracking pause on normal C release; actual20deg/s servo and original motor/contact limits remain. Probe-evidence and bearing evidence still use real actual motion. An exact native-command cache avoids recomputing identical parameter tuples; it rebuilds the same spatial spline and matches uncached q exactly.

Current receiver chart/bearing definition supersedes prior wording: {
  "domain": {
    "origin": "Prior entry is a coordinate chart, not a known physical endpoint",
    "lower": "Known source miter plane intersected with prior cylinder, minus20mm conservative prior margin",
    "upper_axial_m": 1.0,
    "margin_m": 0.02
  },
  "capture": {
    "minimum_loaded_s": 2.0,
    "minimum_commanded_fore_motion_deg": 0.25,
    "minimum_actual_fore_motion_deg": 0.25,
    "minimum_receiving_vertical_fraction": 0.4,
    "chart_wrap_persistence_s": 2.0,
    "geometry_reliability_required": false,
    "front_motion_hinges": 8,
    "front_six_geometry_wrap_required": false
  },
  "head_probe_tracking": [
    0,
    1
  ]
}
Prior cylinder extrapolates to known source cut; no true target endpoint or labels enter control.

Current staged handoff supersedes absolute15s source retirement: {
  "initial_feed_rate_fraction": 0.1,
  "weak_bearing_feed_rate_fraction": 0.05,
  "weak_bearing_weight_fraction": 0.2,
  "strong_bearing_weight_fraction": 0.4,
  "distributed_min_materials": 2,
  "distributed_persistence_s": 2.0,
  "posterior_release_duration_s": 15.0,
  "front_release_materials_after_bearing": 14.0,
  "front_release_rate_materials_s": 0.5,
  "bearing_recovery_timeout_s": 20.0,
  "source_restore_duration_s": 4.0,
  "minimum_each_material_vertical_N": 0.2
}
Actual strong receiving load permits gradual front12to14 release and initial0.1 shared feed. Two positively loaded receiving materials maintained2s permit gradual posterior/native release; source C remains active until this feedback release completes. Lost bearing stops shared S/roll; local C keeps working, with20s recovery timeout.

Native Bishop integration uses algebraically equivalent NumPy rotation prefix products. NATIVE_VECTOR_EQUIVALENCE.json records zero-integration comparisons at S0/9/20, command differences below3e-15rad and unchanged source/target/material/physics choices. Existing actual frozen runs are not modified.


Posterior local C now adjusts hinges12..19 to counter the change of actual gravity plus measured non-source contact X-moment from the initially held equilibrium. Exact robot COM Jacobians are reduced with a measured known-source bearing pose; no receiving mesh, true axis/radius, labels or G overwrite enters it. The local curvature offset is bounded25deg per native joint, rate4deg/s, gain0.7/s, leak0.05/s, and fades15s after measured receiving capture. Saturated drive directions that would increase unavailable torque are excluded. These are registered local-controller choices, not user tolerances. Original physical actuator/control/joint/penetration/support guards remain. This explicitly relaxes the earlier posterior-command lock for necessary source posture adjustment. It never advances source F/M/rolling before capture. A single actual-saved pose/load directional derivative check confirms the analytic moment gradient; it does not prove source bearing or full transfer. SOURCE_BALANCE_GRADIENT_CHECK.json records that limited check.


Current V10 execution supersedes the earlier assumed fixed source frame. Initial global G remains the A16-derived future family. Local C uses only the current robot source boundary, a C2 quintic connector and G receiving helix; no actual body is written into G. A small private finite-rod least-squares correction around native Bishop integrals reduces known mapping errors. Its model strips receiving truth and builds only a synthetic cylinder from applied G, retaining known source and original robot. Geometric position and coarse clearance are soft local costs; no full-path IK/precise-world acceptance or2micrometre command-plane interlock is introduced. Original physical servo and contact/joint/self/floor guards decide actual motion. G writes are bounded using the same-state actual C command preview. C planning runs5Hz, local force/probe and actual servo commands40Hz, original physics1kHz. On validated bearing, C continues on its planned physical route with original feed/rolling/native mapping and finite-rod correction; staged actual-load gates remain. Posterior source-balance C acts only on retained source-role joints, so it does not add a conflicting offset to released fore joints. Actual source pose is allowed C boundary feedback; it is not estimated target geometry or a G overwrite. Small residuals/solver termination are diagnostics, not a dynamics-success claim.
