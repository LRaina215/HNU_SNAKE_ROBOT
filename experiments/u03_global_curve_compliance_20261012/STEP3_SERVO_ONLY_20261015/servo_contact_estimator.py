"""Conservative motor-residual point-force and interaction-axis hypotheses.

Only joint angles, calibrated incremental motor loads, and a virtual G rear
shape are inputs. This is an identifiability diagnostic, not a contact sensor.
Its tail-fixed, quasi-static, single-force model must be explicitly qualified
before any future control use. No simulator/scene state is read by this module.
"""
import numpy as np
from servo_geometry import relative_fk, LINK


DEFAULTS = dict(
    torque_limit_Nm=2.355,
    finite_difference_rad=1e-6,
    torque_noise_sigma_Nm=.02,
    min_increment_norm_Nm=.18,
    max_relative_residual=.18,
    max_rms_residual_Nm=.06,
    min_jacobian_singular_ratio=.015,
    min_force_N=.5,
    max_force_N=60.,
    max_alignment_rms_m=.012,
    max_alignment_error_m=.030,
    min_pose_rms_change_rad=float(np.deg2rad(3.)),
    min_pose_changed_joints=2,
    min_axis_records=5,
    max_axis_records=24,
    min_axis_position_span_m=.035,
    min_normal_middle_eigenvalue=.04,
    max_normal_null_ratio=.18,
    max_axis_normal_rms=.15,
    evidence_max_age_s=30.,
)

ASSUMPTIONS = [
    'tail body pose is fixed while taking the quasi-static joint Jacobian',
    'a single front point force acts at a rod center; shell contact lever arm is omitted',
    'incremental motor torque approximately equals minus Jacobian transpose times incremental force',
    'force direction is a candidate surface normal only if normal force dominates friction',
    'rear measured body shape can be aligned to the current virtual G shape without an external pose',
    'the virtual G rear chart retains a common material gauge across accumulated observations',
]
UNMODELED = ['gravity changes between baseline and sample', 'joint and contact friction',
             'inertia and acceleration', 'rear slip or support redistribution',
             'multiple simultaneous contacts', 'motor-load calibration error',
             'contact point offset from rod center']


def _vector(value, name):
    value = np.asarray(value, float)
    if value.shape != (20,) or not np.all(np.isfinite(value)):
        raise ValueError(name+' must contain twenty finite values')
    return value


def tail_fixed_fk(q):
    """Relative FK expressed with the tail body fixed at the local origin."""
    fk = relative_fk(q)
    rotation = fk['rotations_headframe'][-1]
    origin = fk['nodes_headframe'][-1]
    nodes = (fk['nodes_headframe']-origin)@rotation
    return dict(nodes_tailframe=nodes, centers_tailframe=.5*(nodes[:-1]+nodes[1:]))


def front_jacobians_tail(q, epsilon=1e-6):
    """Return (8,3,20) point Jacobians and the nominal tail-relative nodes."""
    q = _vector(q, 'q')
    nominal = tail_fixed_fk(q)
    jacobian = np.empty((8, 3, 20))
    for joint in range(20):
        plus, minus = q.copy(), q.copy()
        plus[joint] += epsilon
        minus[joint] -= epsilon
        jacobian[:, :, joint] = (tail_fixed_fk(plus)['centers_tailframe'][:8]-
                                 tail_fixed_fk(minus)['centers_tailframe'][:8])/(2*epsilon)
    return jacobian, nominal


def _align_rear(nodes, g_backbone_points):
    target = np.asarray(g_backbone_points, float)
    if target.shape == (22, 3):
        target = target[8:]
    if target.shape != (14, 3) or not np.all(np.isfinite(target)):
        raise ValueError('g_backbone_points must be 22 G nodes or its 14 rear nodes, in virtual chart coordinates')
    source = nodes[8:]
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, singular, vh = np.linalg.svd((source-source_center).T@(target-target_center))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vh.T@u.T)
    rotation = vh.T@correction@u.T
    translation = target_center-rotation@source_center
    errors = np.linalg.norm(source@rotation.T+translation-target, axis=1)
    return rotation, translation, dict(rms_m=float(np.sqrt(np.mean(errors**2))),
                                      max_m=float(np.max(errors)),
                                      shape_singular_values=singular.tolist(),
                                      interpretation='relative rear-shape registration to virtual G; not world localization')


def infer_single(q, load_delta, g_backbone_points, settings=None):
    """Infer or reject a single center-point-force explanation.

    ``load_delta`` is already calibrated to incremental N m by the caller.
    A torque increment alone does not certify that neither original signal was
    saturated; the caller must separately reject saturated baseline/sample data.
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)
    q, torque = _vector(q, 'q'), _vector(load_delta, 'load_delta')
    result = dict(accepted=False, rejection_reasons=[], candidate=None,
                  torque_increment_norm_Nm=float(np.linalg.norm(torque)),
                  material_candidates=[], alignment=None,
                  assumptions=ASSUMPTIONS.copy(), unmodeled_terms=UNMODELED.copy(),
                  target_identity_unresolved=True, measured_contact_point=False,
                  measured_surface_normal=False)
    if np.any(np.abs(torque) >= .98*cfg['torque_limit_Nm']):
        result['rejection_reasons'].append('increment reaches calibrated limit; saturation/linearity cannot be excluded')
    if np.linalg.norm(torque) < cfg['min_increment_norm_Nm']:
        result['rejection_reasons'].append('incremental motor signal is below the identifiable-force threshold')
    if result['rejection_reasons']:
        return result
    jacobians, fk = front_jacobians_tail(q, cfg['finite_difference_rad'])
    rotation, translation, alignment = _align_rear(fk['nodes_tailframe'], g_backbone_points)
    result['alignment'] = alignment
    if alignment['rms_m'] > cfg['max_alignment_rms_m'] or alignment['max_m'] > cfg['max_alignment_error_m']:
        result['rejection_reasons'].append('rear measured shape does not register consistently to the current G rear shape')

    candidates = []
    for material, jacobian in enumerate(jacobians):
        matrix = -jacobian.T
        force, _, rank, singular = np.linalg.lstsq(matrix, torque, rcond=None)
        reconstructed = matrix@force
        residual = torque-reconstructed
        magnitude = float(np.linalg.norm(force))
        candidates.append(dict(material=material, force_tailframe_N=force,
                               position_tailframe_m=fk['centers_tailframe'][material],
                               magnitude_N=magnitude, rank=int(rank),
                               singular_ratio=float(singular[-1]/max(singular[0], 1e-12)),
                               residual_norm_Nm=float(np.linalg.norm(residual)),
                               residual_rms_Nm=float(np.sqrt(np.mean(residual**2))),
                               relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(torque)),
                               reconstructed_increment_Nm=reconstructed))
    candidates.sort(key=lambda item: item['residual_norm_Nm'])
    best = candidates[0]
    result['material_candidates'] = [
        {key: item[key] for key in ('material', 'magnitude_N', 'rank', 'singular_ratio',
                                    'residual_norm_Nm', 'residual_rms_Nm', 'relative_residual')}
        for item in candidates]
    if best['rank'] < 3 or best['singular_ratio'] < cfg['min_jacobian_singular_ratio']:
        result['rejection_reasons'].append('front force Jacobian has insufficient rank or directional observability')
    if best['relative_residual'] > cfg['max_relative_residual'] or best['residual_rms_Nm'] > cfg['max_rms_residual_Nm']:
        result['rejection_reasons'].append('single-point quasi-static force does not explain the motor increments')
    if not cfg['min_force_N'] <= best['magnitude_N'] <= cfg['max_force_N']:
        result['rejection_reasons'].append('fitted force is outside the registered signal-scale bounds')
    ambiguity_limit = best['residual_norm_Nm']+cfg['torque_noise_sigma_Nm']*np.sqrt(20.)
    ambiguous = [item['material'] for item in candidates[1:]
                 if item['residual_norm_Nm'] <= ambiguity_limit and
                 np.linalg.norm(item['position_tailframe_m']-best['position_tailframe_m']) > .35*LINK]
    if ambiguous:
        result['rejection_reasons'].append('different front materials explain the motor increments within the noise allowance')
    result['ambiguous_materials'] = ambiguous
    if result['rejection_reasons']:
        return result
    force_g = rotation@best['force_tailframe_N']
    point_g = rotation@best['position_tailframe_m']+translation
    confidence = float(np.clip(1-best['relative_residual']/cfg['max_relative_residual'], 0., 1.) *
                       np.clip(best['singular_ratio']/(2*cfg['min_jacobian_singular_ratio']), 0., 1.) *
                       np.clip(1-alignment['rms_m']/cfg['max_alignment_rms_m'], 0., 1.))
    result.update(accepted=True,
                  candidate=dict(material=best['material'], point_in_G_m=point_g.tolist(),
                                 force_in_G_N=force_g.tolist(),
                                 normal_candidate_in_G=(force_g/max(np.linalg.norm(force_g), 1e-12)).tolist(),
                                 force_magnitude_N=best['magnitude_N'], confidence=confidence,
                                 relative_residual=best['relative_residual'],
                                 force_tailframe_N=best['force_tailframe_N'].tolist(),
                                 reconstructed_increment_Nm=best['reconstructed_increment_Nm'].tolist(),
                                 interpretation='single-force center hypothesis; normal meaning requires negligible tangential force'))
    return result


class ServoContactEstimator:
    """Accumulate geometrically independent accepted motor-force hypotheses.

    Axis acceptance means consistency under the listed assumptions. It does not
    identify a target pipe or remove ambiguity from source fittings/slip/friction.
    Executing a probe or giving it a new ID never counts as evidence by itself.
    """
    def __init__(self, settings=None):
        self.settings = dict(DEFAULTS)
        if settings:
            self.settings.update(settings)
        self.records = []
        self.last = None

    def observe(self, q, load_delta, g_backbone_points, *, t=None, probe_id=None):
        q = _vector(q, 'q')
        result = infer_single(q, load_delta, g_backbone_points, self.settings)
        if t is not None:
            t = float(t)
            if not np.isfinite(t):
                raise ValueError('time must be finite')
            self.records = [record for record in self.records
                            if record['t'] is None or t-record['t'] <= self.settings['evidence_max_age_s']]
        independent = bool(result['accepted'])
        if independent:
            for record in self.records:
                difference = np.abs(q-record['q'])
                if np.sqrt(np.mean(difference**2)) < self.settings['min_pose_rms_change_rad'] or \
                        np.count_nonzero(difference >= self.settings['min_pose_rms_change_rad']) < self.settings['min_pose_changed_joints']:
                    independent = False
                    break
        if independent:
            self.records.append(dict(q=q.copy(), t=t, probe_id=probe_id, **result['candidate']))
            self.records = self.records[-self.settings['max_axis_records']:]
        result.update(independent_pose_added=independent, independent_record_count=len(self.records),
                      axis_candidate=None, axis_confidence=0., axis_rejection_reasons=[])
        if len(self.records) < self.settings['min_axis_records']:
            result['axis_rejection_reasons'].append('too few independent accepted force/pose hypotheses')
        elif not result['accepted']:
            result['axis_rejection_reasons'].append('current force evidence is rejected; historical axis is not emitted as currently reliable')
        else:
            normals = np.asarray([record['normal_candidate_in_G'] for record in self.records])
            points = np.asarray([record['point_in_G_m'] for record in self.records])
            values, vectors = np.linalg.eigh(normals.T@normals/len(normals))
            axis = vectors[:, 0]
            if axis[np.argmax(np.abs(axis))] < 0:
                axis = -axis
            span = float(np.max(np.linalg.norm(points[:, None]-points[None, :], axis=2)))
            null_ratio = float(values[0]/max(values[1], 1e-12))
            normal_rms = float(np.sqrt(np.mean((normals@axis)**2)))
            result['axis_diagnostics'] = dict(normal_eigenvalues=values.tolist(),
                                              normal_null_ratio=null_ratio, normal_axis_rms=normal_rms,
                                              position_span_m=span,
                                              distinct_probe_ids=len(set(str(record['probe_id']) for record in self.records)))
            if values[1] < self.settings['min_normal_middle_eigenvalue']:
                result['axis_rejection_reasons'].append('accepted force directions lack independent angular coverage')
            if null_ratio > self.settings['max_normal_null_ratio'] or normal_rms > self.settings['max_axis_normal_rms']:
                result['axis_rejection_reasons'].append('accepted force directions do not support one consistently orthogonal axis')
            if span < self.settings['min_axis_position_span_m']:
                result['axis_rejection_reasons'].append('accepted center hypotheses have insufficient spatial separation')
            if not result['axis_rejection_reasons']:
                confidence = np.mean([record['confidence'] for record in self.records]) * \
                    np.clip(1-null_ratio/self.settings['max_normal_null_ratio'], 0., 1.) * \
                    np.clip(values[1]/(.12), 0., 1.)
                result.update(axis_candidate=axis.tolist(), axis_confidence=float(confidence),
                              axis_sign_unobservable=True, radius_identified=False,
                              interpretation='unoriented interaction-axis hypothesis from motor residuals under tail-fixed normal-force assumptions')
        self.last = result
        return result
