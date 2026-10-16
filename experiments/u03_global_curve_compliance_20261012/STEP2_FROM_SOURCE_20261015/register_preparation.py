"""Write the from-source preregistration. Does not launch experiments."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;old=P.parent/'STEP2_20261014'
previous=json.loads((old/'REGISTERED.json').read_text());proof=json.loads((P/'SOURCE_INITIAL_PROOF.json').read_text())
reg={k:previous[k] for k in ['conditions','cases','source_known','estimated_contact_classes','estimator','probe','runtime']}
reg.update(version='FROM_SOURCE_R1_PREPARATION',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,
    scope='One full-source t0, all21 rods; continuous source stabilization, release, extension, contact refinement, acquisition, full transfer and sustained roll',
    source_constants=json.loads((P/'source_constants.json').read_text()),robot_weight_N=proof['robot_weight_N'],initial_file='SOURCE_INITIAL.npz',
    initial_definition='U03 t0 coil with one prescribed source-X placement and static compliant adjustment; no saved intermediate action state; all physics unchanged',
    initial_static_equilibrium_exact=False,initial_stabilization_is_logged_from_t0=True,
    duration_s=235.,nominal_frequency_hz=.02,command_interval_s=.025,servo_slew_deg_s=20.,servo_limit_deg=88.,
    support=dict(initial_load_min_fraction=.90,initial_load_max_fraction=1.10,initial_velocity_max=.03,initial_stable_s=2.,initial_settle_timeout_s=10.,
        rear_load_min_fraction_before_capture=.40,receiver_load_min_fraction_after_capture=.40,rear_low_load_max_s=.25),
    global_update=dict(previous['global_update'],theta_min_coherence=.70,theta_bound_deg=15.,theta_rate_deg_s=.25,
        G_position_step_m=.0004,G_tangent_step_deg=.10,G_curvature_step_per_m=.02,material_frame_step_deg=.10),
    local_feedback=dict(receiving_delta_bound_m=.002,receiving_delta_rate_m_s=.00025,source_gamma_bound_per_m=.4,source_gamma_rate_per_m_s=.08,
        source_relief_enabled_before_capture=False,source_relief_requires_confirmed_front_and_current_receiver_bearing=True,front_relief_bound_per_m=.30,front_relief_rate_per_m_s=.06,front_force_threshold_N=12.,front_force_gain_per_m_N=.012,
        always_executed=True,initial_native_curve_preload_fade_S=.75),
    probe_trigger='First front material0..7 contact inconsistent with known source, before cylinder qualification; two4s signed cycles at0.18/m',
    probe_duration_reason='Acquire both probe directions during the initial contact-to-wrap interval; old late-start14s directions do not protect this ordering. New source-start preregistration before any dynamics.',
    theta_estimation='Circular mean of contact polar angle minus2pi*axial/pitch; helix-contact hypothesis; coherence>=.70; bounded offset, no physical inlet plane identification',
    input_boundary='Robot poses/joint telemetry and unlabeled contact point/normal/full world force/signed gap; known finite source; rough receiver prior. No receiver/elbow labels, true target axis/radius/mesh into controller.',
    common_pair='Exact same SOURCE_INITIAL bytes, code, prior, observation rules, probe trigger, local feedback, frequency, budget and support checks; only G write permission differs. Physical measurements may diverge after writeback.',
    all_source_start_gate='Offline truth:all21 source envelope and target/elbow zero contact; online:all21 known source geometry and source load90..110%weight, allqvelmax<.03 for2s before any S advance. Full stabilization remains in logged t0 timeline.',
    rear_support_policy='Known-source and continuous source/unknown rear tracks exclude qualified receiver. Before capture or whenever retained front geometry/current qualified receiver bearing>=40%weight is lost, rear bearing<40%weight for.25s stops/saves failure. Capture confirmation alone never disables support checks.',
    acceptance=dict(previous['acceptance'],source_initial_21=True,no_pre_acquired_target_wrap=True,
        initial_full_source_stabilization_required=True,front_capture_must_be_new_after_contact=True,
        simultaneous_source_target_bearing_fraction=.20,simultaneous_bearing_s=.5,minimum_rear_source_roll_during_acquisition_deg=15.,
        required_chain=['full-source t0','logged full-source stabilization','curve-driven release and extension','raw front contact','two-direction contact evidence','reliable contact geometry','G write in enabled cases','new front wrap and bearing','all21 transfer','50s+ rolling']),
    failure_policy='Record every started case, fail/save if stabilization/acquisition/support/physical limits fail; no replacement later start or pose replay. Geometry-only success is insufficient.',
    first_run_policy='No dynamics before independent source-initial and actual-code-chain review; runner enforces registration plus INDEPENDENT_REVIEW.json approval.',
    new_dynamics_starts=0)
reg['probe']['direction_duration_s']=4.
reg['probe']['directions']='Two4s signed sinusoidal curvature cycles on material0..8, triggered by raw front non-source contact; no active precontact search'
(P/'REGISTERED.json').write_text(json.dumps(reg,indent=2)+'\n')
print(json.dumps(dict(version=reg['version'],dynamics_authorized=False,source_stabilization_required=True,cases=len(reg['cases'])),indent=2))
