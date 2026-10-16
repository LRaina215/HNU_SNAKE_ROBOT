"""Unknown cylinder hypotheses from loaded, unlabeled surface observations.

``Geometry.entry`` is an evidence-derived reference station on an infinite
cylinder axis. Side contacts do not identify a physical inlet or cylinder end.
Task direction only chooses the axis sign; it supplies no center or radius.
"""
import numpy as np
from scipy.optimize import least_squares

from contact_sensor import known_source
from geometry import Geometry, normalize


class UnknownCylinderEstimator:
    def __init__(self, source, settings, task_direction):
        self.source = source
        self.settings = settings
        direction = np.asarray(task_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-9:
            raise ValueError('task_direction must be a finite nonzero vector')
        self.task_direction = normalize(direction)
        bounds = settings.get('physical_radius_bounds_m', settings.get('radius_bounds_m',
                              [settings.get('radius_min_m', .025), settings.get('radius_max_m', .12)]))
        self.radius_bounds = np.asarray(bounds, dtype=float)
        if self.radius_bounds.shape != (2,) or not np.all(np.isfinite(self.radius_bounds)) or not 0 < self.radius_bounds[0] < self.radius_bounds[1]:
            raise ValueError('physical radius bounds must be positive and increasing')
        self.geometry = None
        self.samples = []
        self._observations = []
        self._reference_station_point = None
        self.next_fit = 0.
        self.first_front_contact = None
        self.first_contact = None
        self.first_reliable = None
        self.probe_seen = np.zeros(2, dtype=bool)
        self.fit_history = []
        self.last = self._result(0., [], 'no evidence')

    def _result(self, t, samples, reason):
        return dict(t=float(t), reliable=False, confidence=0., reason=reason,
                    sample_count=len(samples), independent_point_count=len(samples),
                    material_ids=sorted(set(x['material'] for x in samples)),
                    probe_seen=self.probe_seen.tolist(),
                    latest_evidence_t=max((x['t'] for x in samples), default=None),
                    axial_entry_observable=False, axial_entry_estimated=False,
                    axial_entry_uncertainty_m=None,
                    reference_station_kind='evidence projection, not physical inlet',
                    physical_radius_bounds_m=self.radius_bounds.tolist())

    def observe(self, sensor, t, probe):
        self.probe_seen |= np.abs(np.asarray(probe)) > .04
        minimum_load = float(self.settings.get('minimum_normal_force_N', .02))
        new = []
        for contact in sensor['contacts']:
            point = np.asarray(contact['point'], dtype=float)
            normal = np.asarray(contact['normal'], dtype=float)
            force = float(contact['normal_N'])
            if point.shape != (3,) or normal.shape != (3,) or not np.all(np.isfinite(point)) or not np.all(np.isfinite(normal)):
                continue
            if not np.isfinite(force) or force < minimum_load or np.linalg.norm(normal) < 1e-8:
                continue
            clean = dict(contact, point=point, normal=normalize(normal))
            if known_source(clean, self.source):
                continue
            material = int(contact['material'])
            if not 0 <= material < 21:
                continue
            if self.first_contact is None:
                self.first_contact = float(t)
            if material < 8 and self.first_front_contact is None:
                self.first_front_contact = float(t)
            item = dict(t=float(t), material=material, point=point.tolist(),
                        normal=clean['normal'].tolist(), normal_N=force,
                        penetration_m=float(contact.get('penetration_m', 0.)),
                        source_excluded=False, candidate_gate=True)
            self._observations.append(item)
            new.append(item)
        cutoff = t - float(self.settings.get('window_s', 12.))
        self._observations = [x for x in self._observations if x['t'] >= cutoff]
        self.samples = [x for x in self.samples if x['t'] >= cutoff]
        # A cached fit cannot remain reliable after its own supporting evidence
        # expires, even when the next scheduled fit has not arrived yet.
        expired_fit = bool(self.last.get('reliable', False) and
                           self.last.get('oldest_evidence_t', -np.inf) < cutoff)
        if t >= self.next_fit or expired_fit:
            self.next_fit = t + float(self.settings.get('fit_interval_s', 1.))
            self.fit(t)
        return new

    def _unique_samples(self):
        point_cell = float(self.settings.get('independent_point_cell_m', .003))
        normal_cell = float(self.settings.get('independent_normal_cell', .08))
        unique = {}
        for item in self._observations:
            key = tuple(np.rint(np.asarray(item['point']) / point_cell).astype(int))
            key += tuple(np.rint(np.asarray(item['normal']) / normal_cell).astype(int))
            unique[key] = item
        samples = sorted(unique.values(), key=lambda x: x['t'])
        limit = max(24, int(self.settings.get('maximum_fit_points', 256)))
        if len(samples) > limit:
            samples = [samples[i] for i in np.linspace(0, len(samples) - 1, limit, dtype=int)]
        return samples

    def _signed_axis(self, axis):
        axis = normalize(axis)
        return -axis if axis @ self.task_direction < 0 else axis

    @staticmethod
    def _errors(geometry, points, normals):
        _, radial, rho = geometry.coordinates(points)
        radial_normal = radial / np.maximum(rho[:, None], 1e-12)
        return (np.abs(rho - geometry.radius),
                np.abs(normals @ geometry.axis),
                np.linalg.norm(radial_normal - normals, axis=1))

    def _inliers(self, geometry, points, normals):
        distance, axial, alignment = self._errors(geometry, points, normals)
        return ((distance < float(self.settings.get('inlier_point_residual_m', .0025))) &
                (axial < float(self.settings.get('inlier_normal_axis_residual', .15))) &
                (alignment < float(self.settings.get('inlier_normal_alignment', .18))))

    def _seed(self, points, normals):
        # Pair-normal cross products provide robust null-space seeds without
        # assuming a target axis. Each candidate is scored against every point.
        n = len(points)
        rng = np.random.default_rng(0)
        _, vectors = np.linalg.eigh(normals.T @ normals / n)
        axes = [vectors[:, 0]]
        for _ in range(int(self.settings.get('axis_seed_pairs', 48))):
            i, j = rng.choice(n, 2, replace=False)
            cross = np.cross(normals[i], normals[j])
            if np.linalg.norm(cross) > .2:
                axes.append(normalize(cross))
        best = None
        best_score = (-1, -np.inf)
        for axis in axes:
            axis = self._signed_axis(axis)
            aligned = np.flatnonzero(np.abs(normals @ axis) < .2)
            if len(aligned) < 4:
                continue
            e, b = Geometry(np.zeros(3), axis, 1.).frame()
            basis = np.column_stack([e, b])
            uv, nv = points @ basis, normals @ basis
            # In the transverse plane, p_i = center + radius * normal_i.
            # Include a full linear fit and robust two-observation hypotheses.
            matrix = np.zeros((2 * len(aligned), 3))
            matrix[::2, 0] = 1.
            matrix[1::2, 1] = 1.
            matrix[::2, 2] = nv[aligned, 0]
            matrix[1::2, 2] = nv[aligned, 1]
            candidates = [np.linalg.lstsq(matrix, uv[aligned].ravel(), rcond=None)[0]]
            for _ in range(int(self.settings.get('center_seed_pairs', 12))):
                i, j = rng.choice(aligned, 2, replace=False)
                dn = nv[i] - nv[j]
                if dn @ dn < .04:
                    continue
                radius = float((uv[i] - uv[j]) @ dn / (dn @ dn))
                center = .5 * (uv[i] + uv[j] - radius * (nv[i] + nv[j]))
                candidates.append(np.r_[center, radius])
            for candidate in candidates:
                if not self.radius_bounds[0] < candidate[2] < self.radius_bounds[1]:
                    continue
                center = basis @ candidate[:2] + axis * np.mean(points[aligned] @ axis)
                geometry = Geometry(center, axis, candidate[2])
                mask = self._inliers(geometry, points, normals)
                distance, _, alignment = self._errors(geometry, points, normals)
                score = (int(mask.sum()), -float(np.mean(distance[mask] + .01 * alignment[mask])) if mask.any() else -np.inf)
                if score > best_score:
                    best, best_score = (geometry, mask), score
        return best

    def fit(self, t):
        samples = self._unique_samples()
        self.samples = samples
        result = self._result(t, samples, 'insufficient distinct evidence')
        minimum_points = int(self.settings.get('minimum_distinct_samples', 12))
        minimum_materials = int(self.settings.get('minimum_materials', 2))
        if len(samples) < minimum_points or len(result['material_ids']) < minimum_materials:
            self.last = result
            self.fit_history.append(result)
            return
        points = np.asarray([x['point'] for x in samples])
        normals = np.asarray([x['normal'] for x in samples])
        seed = self._seed(points, normals)
        if seed is None or seed[1].sum() < max(4, minimum_points // 2):
            result['reason'] = 'no supported cylinder within broad physical radius bounds'
            self.last = result
            self.fit_history.append(result)
            return
        geometry, mask = seed
        # Re-estimate the smallest normal eigenvector on the robust consensus;
        # unlike a full-cloud eigenvector this resists elbow/cap outliers.
        for _ in range(2):
            if mask.sum() < 4:
                break
            selected_points, selected_normals = points[mask], normals[mask]
            _, eigenvectors = np.linalg.eigh(selected_normals.T @ selected_normals / mask.sum())
            axis = self._signed_axis(eigenvectors[:, 0])
            e, b = Geometry(geometry.entry, axis, geometry.radius).frame()
            station = (np.mean(selected_points, axis=0) if self._reference_station_point is None
                       else self._reference_station_point)
            base = geometry.entry.copy()

            def candidate(v):
                a = normalize(axis + v[0] * e + v[1] * b)
                center = base + v[2] * e + v[3] * b
                center += a * ((station - center) @ a)
                return Geometry(center, a, v[4])

            def residual(v):
                g = candidate(v)
                _, radial, rho = g.coordinates(selected_points)
                rn = radial / np.maximum(rho[:, None], 1e-12)
                return np.r_[rho - g.radius, .025 * (selected_normals @ g.axis),
                             .012 * (rn - selected_normals).ravel()]

            lower = [-1., -1., -np.inf, -np.inf, self.radius_bounds[0]]
            upper = [1., 1., np.inf, np.inf, self.radius_bounds[1]]
            start = [0., 0., 0., 0., np.clip(geometry.radius, self.radius_bounds[0] + 1e-8, self.radius_bounds[1] - 1e-8)]
            fit = least_squares(residual, start, bounds=(lower, upper), loss='soft_l1',
                                f_scale=.001, max_nfev=40)
            geometry = candidate(fit.x)
            mask = self._inliers(geometry, points, normals)
        count = int(mask.sum())
        if count == 0:
            result['reason'] = 'cylinder refinement lost all inliers'
            self.last = result
            self.fit_history.append(result)
            return
        inlier_samples = [sample for sample, valid in zip(samples, mask) if valid]
        values = np.linalg.eigvalsh(normals[mask].T @ normals[mask] / count)
        distance, axial, alignment = self._errors(geometry, points[mask], normals[mask])
        point_rms = float(np.sqrt(np.mean(distance ** 2)))
        axis_rms = float(np.sqrt(np.mean(axial ** 2)))
        normal_rms = float(np.sqrt(np.mean(alignment ** 2)))
        axial_span = float(np.ptp(points[mask] @ geometry.axis))
        materials = sorted(set(x['material'] for x in inlier_samples))
        min_diversity = float(self.settings.get('minimum_middle_normal_eigenvalue', .08))
        min_span = float(self.settings.get('minimum_axial_span_m', .035))
        max_point = float(self.settings.get('maximum_point_rms_m', .0015))
        max_axis = float(self.settings.get('maximum_normal_axis_rms', .08))
        max_normal = float(self.settings.get('maximum_normal_alignment_rms', .12))
        min_fraction = float(self.settings.get('minimum_inlier_fraction', .65))
        checks = dict(independent_points=count >= minimum_points,
                      material_diversity=len(materials) >= minimum_materials,
                      normal_diversity=values[1] >= min_diversity,
                      axial_span=axial_span >= min_span,
                      inlier_fraction=count / len(samples) >= min_fraction,
                      point_residual=point_rms <= max_point,
                      normal_axis_residual=axis_rms <= max_axis,
                      normal_alignment=normal_rms <= max_normal)
        checks = {key: bool(value) for key, value in checks.items()}
        reliable = bool(all(checks.values()))
        # These scores are conditioning/residual diagnostics, not calibrated CIs.
        quality = max(0., 1. - max(point_rms / max_point, axis_rms / max_axis, normal_rms / max_normal))
        strength = min(1., count / minimum_points, values[1] / min_diversity, axial_span / min_span)
        result.update(reliable=reliable, confidence=float(count / len(samples) * quality * strength) if reliable else 0.,
                      reason='qualified unknown cylinder' if reliable else 'failed: ' + ', '.join(k for k, v in checks.items() if not v),
                      candidate=geometry.as_dict(), qualification_checks=checks,
                      normal_spectrum=values.tolist(), axial_span_m=axial_span,
                      point_rms_m=point_rms, normal_axis_rms=axis_rms,
                      normal_alignment_rms=normal_rms, inlier_count=count,
                      inlier_fraction=float(count / len(samples)), inlier_material_ids=materials,
                      latest_evidence_t=max(x['t'] for x in inlier_samples),
                      oldest_evidence_t=min(x['t'] for x in inlier_samples),
                      task_direction_axis_angle_deg=float(np.degrees(np.arccos(np.clip(geometry.axis @ self.task_direction, -1., 1.)))))
        if reliable:
            if self._reference_station_point is None:
                self._reference_station_point = np.mean(points[mask], axis=0)
            geometry.entry += geometry.axis * ((self._reference_station_point - geometry.entry) @ geometry.axis)
            result['candidate'] = geometry.as_dict()
            result['reference_station_evidence_point_m'] = self._reference_station_point.tolist()
            self.geometry = geometry
            # GlobalUpdater may consume these points for phase estimation: never
            # expose discarded elbow/cap contacts as qualified cylinder samples.
            self.samples = inlier_samples
            if self.first_reliable is None:
                self.first_reliable = float(t)
        self.last = result
        self.fit_history.append(result)
