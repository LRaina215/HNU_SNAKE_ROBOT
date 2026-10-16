"""Body-relative shape evidence from twenty joint angles and load increments.

There is deliberately no simulator or external-pose interface. A fitted
cylinder describes the rear body's shape; it neither proves contact with a
pipe nor identifies a pipe surface or endpoint. Load novelty cannot uniquely
separate a new pipe, a source fitting, gravity, self-contact, or model error.
"""
import numpy as np
from scipy.optimize import least_squares

LINK = .0645
_RZ_PI = np.diag([-1., -1., 1.])


def _angles(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (20,) or not np.all(np.isfinite(q)):
        raise ValueError('q must contain twenty finite measured servo angles')
    return q


def relative_fk(q):
    """Return 22 backbone nodes in the current head body's local frame.

    The head body origin is node 1. Node 0 is its forward endpoint. Neither
    floating-base translation nor floating-base orientation is an input.
    """
    q = _angles(q)
    origin, rotation = np.zeros(3), np.eye(3)
    nodes = [np.array([0., 0., LINK]), origin.copy()]
    rotations = [rotation.copy()]
    for index, angle in enumerate(q):
        c, s = np.cos(angle), np.sin(angle)
        if index % 2 == 0:  # joint 1,3,...: child-frame +X
            joint_rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
        else:              # joint 2,4,...: child-frame -Y
            joint_rotation = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])
        rotation = rotation @ _RZ_PI @ joint_rotation
        origin = origin-LINK*rotation[:, 2]
        nodes.append(origin.copy())
        rotations.append(rotation.copy())
    nodes = np.asarray(nodes)
    return dict(nodes_headframe=nodes, centers_headframe=.5*(nodes[:-1]+nodes[1:]),
                rotations_headframe=np.asarray(rotations),
                coordinate_system='measured-angle relative head-body frame')


def _unit(vector):
    return vector/max(float(np.linalg.norm(vector)), 1e-12)


def _basis(axis):
    first = np.eye(3)[np.argmin(np.abs(axis))]
    first = _unit(first-np.dot(first, axis)*axis)
    return np.column_stack([first, np.cross(axis, first)])


def fit_source(q):
    """Fit a rear-body cylindrical-shape hypothesis using centers M8..M20.

    Three PCA axes supply bounded local fits, not scene directions. Radius is
    the *body centerline wrapping radius*. A low-confidence result intentionally
    returns no usable axis/center/radius. This function is stateless, so a caller
    should apply temporal qualification before using the hypothesis.
    """
    fk = relative_fk(q)
    points = fk['centers_headframe'][8:21]
    centroid = points.mean(axis=0)
    centered = points-centroid
    covariance = centered.T@centered/len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    result = dict(reliable=False, confidence=0., axis_headframe=None,
                  axis_point_headframe=None, body_wrap_radius_m=None,
                  coordinate_system='measured-angle relative head-body frame',
                  interpretation='rear-body cylindrical-shape hypothesis, not verified pipe/contact geometry',
                  materials=list(range(8, 21)), pca_eigenvalues_m2=eigenvalues.tolist(),
                  rms_radial_residual_m=None, p95_radial_residual_m=None,
                  circular_coverage_rad=0., axial_span_m=0., observability_ratio=0.,
                  ambiguity=False, rejection_reasons=[])
    if eigenvalues[1] < 1e-5 or eigenvalues[0] < 1e-6:
        result['rejection_reasons'] = ['rear body lacks three-dimensional wrapping evidence']
        return result

    candidates = []
    # No initial-world pose, nominal target direction, or known pipe radius is
    # used. These broad bounds limit only the body's plausible geometric scale.
    direction_bound = np.tan(np.deg2rad(35.))
    for seed in eigenvectors.T[::-1]:
        basis = _basis(seed)
        radial = centered-(centered@seed)[:, None]*seed
        r0 = float(np.clip(np.median(np.linalg.norm(radial, axis=1)), .0251, .2999))

        def unpack(parameters):
            axis = _unit(seed+basis@parameters[:2])
            center = centroid+basis@parameters[2:4]
            return axis, center, parameters[4]

        def residual(parameters):
            axis, center, radius = unpack(parameters)
            delta = points-center
            return np.linalg.norm(delta-(delta@axis)[:, None]*axis, axis=1)-radius

        solution = least_squares(residual, [0., 0., 0., 0., r0],
                                 bounds=([-direction_bound, -direction_bound, -.2, -.2, .025],
                                         [direction_bound, direction_bound, .2, .2, .3]),
                                 loss='soft_l1', f_scale=.003, max_nfev=80,
                                 ftol=1e-7, xtol=1e-7, gtol=1e-7)
        axis, center, radius = unpack(solution.x)
        if np.dot(axis, points[0]-points[-1]) < 0:
            axis = -axis
        center = center-axis*np.dot(center, axis)  # unique closest axis point to head origin
        errors = residual(solution.x)
        singular = np.linalg.svd(solution.jac, compute_uv=False)
        candidates.append(dict(axis=axis, center=center, radius=float(radius),
                               rms=float(np.sqrt(np.mean(errors**2))),
                               p95=float(np.percentile(np.abs(errors), 95)),
                               observability=float(singular[-1]/max(singular[0], 1e-12)),
                               optimality=float(solution.optimality), solver_success=bool(solution.success)))
    candidates.sort(key=lambda x: x['rms'])
    best = candidates[0]
    axis, center = best['axis'], best['center']
    delta = points-center
    axial = delta@axis
    transverse = delta-axial[:, None]*axis
    basis = _basis(axis)
    angles = np.mod(np.arctan2(transverse@basis[:, 1], transverse@basis[:, 0]), 2*np.pi)
    angles = np.sort(angles)
    coverage = float(2*np.pi-np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
    axial_span = float(np.ptp(axial))
    ambiguous = any(np.arccos(np.clip(abs(np.dot(axis, alternative['axis'])), 0., 1.)) > np.deg2rad(20.)
                    and alternative['rms'] <= best['rms']+.002 for alternative in candidates[1:])
    reasons = []
    if best['rms'] > .008 or best['p95'] > .014:
        reasons.append('rear shape has excessive cylindrical-fit residual')
    if coverage < np.pi:
        reasons.append('rear angular coverage is insufficient')
    if axial_span < .10:
        reasons.append('rear axial extent is insufficient to identify an axis')
    if best['observability'] < .002:
        reasons.append('cylindrical-fit Jacobian is poorly conditioned')
    if ambiguous:
        reasons.append('distinct axis hypotheses explain the shape similarly')
    if best['radius'] <= .0251 or best['radius'] >= .2999:
        reasons.append('body-wrap radius reached a broad geometric search bound')
    if not best['solver_success']:
        reasons.append('bounded local fit did not converge')
    confidence = float(np.clip(1-best['rms']/.012, 0., 1.) *
                       np.clip(coverage/np.pi, 0., 1.) *
                       np.clip(axial_span/.16, 0., 1.) *
                       np.clip(best['observability']/.01, 0., 1.))
    if ambiguous:
        confidence *= .25
    result.update(reliable=not reasons, confidence=confidence,
                  rms_radial_residual_m=best['rms'], p95_radial_residual_m=best['p95'],
                  circular_coverage_rad=coverage, axial_span_m=axial_span,
                  observability_ratio=best['observability'], ambiguity=ambiguous,
                  rejection_reasons=reasons,
                  candidate_rms_m=[item['rms'] for item in candidates])
    if not reasons:
        result.update(axis_headframe=axis.tolist(), axis_point_headframe=center.tolist(),
                      body_wrap_radius_m=best['radius'])
    return result


def infer_front_novelty(q, load_delta, source_fit=None, *, load_threshold=.15, departure_margin_m=.025):
    """Conservative joint-load/relative-shape evidence, never contact labeling.

    ``load_delta`` is signed change in the twenty calibrated servo-load signals
    relative to the caller's baseline. Here the default threshold is a fraction
    of rated servo load; current or device units need a calibrated conversion.
    Joint load transmission does not localize a unique point of contact.
    """
    load_delta = np.asarray(load_delta, float)
    if load_delta.shape != (20,) or not np.all(np.isfinite(load_delta)):
        raise ValueError('load_delta must contain twenty finite servo-load increments')
    source_fit = fit_source(q) if source_fit is None else source_fit
    result = dict(possible_front_interaction_beyond_source_envelope=False, score=0.,
                  source_envelope_compatible=None, source_hypothesis_reliable=bool(source_fit['reliable']),
                  front_increment_active_count=int(np.sum(np.abs(load_delta[:8]) >= load_threshold)),
                  front_increment_peak=float(np.max(np.abs(load_delta[:8]))),
                  front_increment_fraction=float(np.sum(np.abs(load_delta[:8]))/max(np.sum(np.abs(load_delta)), 1e-12)),
                  front_radial_departure_m=None, outside_front_materials=[],
                  interpretation='possible front interaction beyond a rear-body envelope; not a target-contact classification',
                  unresolved_alternatives=['source junction/end fitting', 'self contact', 'gravity/load redistribution', 'model/calibration error'],
                  source_fit=source_fit)
    if not source_fit['reliable']:
        return result
    centers = relative_fk(q)['centers_headframe']
    axis = np.asarray(source_fit['axis_headframe'])
    center = np.asarray(source_fit['axis_point_headframe'])
    delta = centers[:8]-center
    radial_distance = np.linalg.norm(delta-(delta@axis)[:, None]*axis, axis=1)
    departure = radial_distance-source_fit['body_wrap_radius_m']
    margin = max(float(departure_margin_m), 3*float(source_fit['p95_radial_residual_m']))
    outside = np.flatnonzero(departure > margin)
    result.update(front_radial_departure_m=departure.tolist(),
                  radial_departure_margin_m=margin, outside_front_materials=outside.tolist(),
                  source_envelope_compatible=bool(len(outside) == 0))
    load_evidence = result['front_increment_active_count'] >= 1 and result['front_increment_fraction'] >= .4
    geometry_evidence = len(outside) >= 2
    possible = bool(load_evidence and geometry_evidence)
    result['possible_front_interaction_beyond_source_envelope'] = possible
    if possible:
        result['score'] = float(source_fit['confidence'] *
                                np.clip(result['front_increment_peak']/(2*load_threshold), 0., 1.) *
                                np.clip((np.max(departure)-margin)/margin, 0., 1.))
    return result
