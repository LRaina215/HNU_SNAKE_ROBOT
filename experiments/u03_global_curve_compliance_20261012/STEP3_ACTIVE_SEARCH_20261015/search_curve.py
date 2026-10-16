"""Evidence-free task-side exploration, then bounded contact-defined G morphing.

Before a cylinder is observed the front guide is an open line/connector derived
from known source endpoint, body link length and coarse task side. No receiving
cylinder placeholder participates in its positions or derivatives.
"""
import numpy as np
from from_source_curve import FromSourceCurve
from estimated_curve import EstimatedCurve
from geometry import Geometry, normalize
from native_mapping import LINK


class ActiveSearchCurve(FromSourceCurve):
    def __init__(self, constants, placeholder, source, task_direction, settings,
                 search_enabled=True):
        self.source_known = source
        self.task_direction = normalize(np.asarray(task_direction, float))
        self.search_settings = settings
        self.search_enabled = bool(search_enabled)
        self.release_fraction = 0.
        self.search_offset = np.zeros(2)
        self.capture_blend = 0.
        self.contact_geometry_installed = False
        self.morph_history = []
        xaxis = np.array([1., 0., 0.])
        lateral = xaxis - self.task_direction * (xaxis @ self.task_direction)
        self.search_basis = np.column_stack([normalize(lateral),
                                             normalize(np.cross(lateral, self.task_direction))])
        endpoint = np.array([source['axial_bounds_m'][1], *source['center_yz_m']], float)
        self.search_station = (endpoint + (source['radius_m'] + LINK) * xaxis
                               + 3 * LINK * self.task_direction
                               - (source['radius_m'] + .5 * LINK) * self.search_basis[:, 1])
        super().__init__(constants, placeholder)

    def growth(self, S):
        return min(float(self.smooth(S / 6.)), self.release_fraction)

    def target_jet(self, u, order=0):
        u = np.asarray(u)
        if order == 0:
            free = self.search_station + self.search_basis @ self.search_offset + u[..., None] * self.task_direction
        elif order == 1:
            free = np.broadcast_to(self.task_direction, u.shape + (3,)).copy()
        else:
            free = np.zeros(u.shape + (3,))
        if self.capture_blend <= 0.:
            return free
        receiving = EstimatedCurve.target_jet(self, u, order)
        return (1 - self.capture_blend) * free + self.capture_blend * receiving

    def set_search(self, requested, dt, release_fraction, support_ok):
        desired = np.asarray(requested, float) if self.search_enabled and support_ok else np.zeros(2)
        step = desired - self.search_offset
        bound = self.search_settings['offset_rate_m_s'] * dt
        step *= min(1., bound / max(float(np.linalg.norm(step)), 1e-12))
        self.search_offset += step
        self.release_fraction = float(release_fraction)

    def install_contact_geometry(self, geometry, S, t):
        if self.contact_geometry_installed:
            return
        q0, i0 = self.command(S)
        old = self.geometry.copy()
        self.geometry = geometry.copy()
        q1, i1 = self.command(S)
        command_delta = float(np.max(abs(q1 - q0)))
        node_delta = float(np.max(np.linalg.norm(i1['global_reference_points'] - i0['global_reference_points'], axis=1)))
        if command_delta > 1e-10 or node_delta > 1e-10:
            self.geometry = old
            raise ValueError('Inactive receiving geometry leaked into precontact G')
        self.contact_geometry_installed = True
        self.morph_history.append(dict(t=float(t), kind='candidate_install_no_G_change',
                                       command_delta_deg=np.degrees(command_delta), node_delta_m=node_delta,
                                       candidate=geometry.as_dict()))

    def advance_contact_morph(self, S, t, dt, qualified):
        if not self.contact_geometry_installed or not qualified or self.capture_blend >= 1.:
            return
        before = self.capture_blend
        q0, i0 = self.command(S)
        p0 = i0['global_reference_points']
        tang0 = np.diff(p0, axis=0)
        tang0 /= np.maximum(np.linalg.norm(tang0, axis=1)[:, None], 1e-12)
        increment = min(1 - before, self.search_settings['contact_morph_rate_s'] * dt)
        accepted = False
        metrics = {}
        for attempt in range(8):
            self.capture_blend = before + increment
            try:
                q1, i1 = self.command(S)
                p1 = i1['global_reference_points']
                tang1 = np.diff(p1, axis=0)
                tang1 /= np.maximum(np.linalg.norm(tang1, axis=1)[:, None], 1e-12)
                metrics = dict(command_step_deg=float(np.max(abs(q1-q0))*180/np.pi),
                               node_step_m=float(np.max(np.linalg.norm(p1-p0,axis=1))),
                               tangent_step_deg=float(np.max(np.arccos(np.clip(np.sum(tang0*tang1,axis=1),-1,1)))*180/np.pi),
                               curvature_component_step_per_m=float(np.max(abs(i1['curvature_field']-i0['curvature_field']))*1000))
                accepted = bool(np.all(np.isfinite(q1))
                    and metrics['command_step_deg'] <= self.search_settings['morph_command_step_deg']
                    and metrics['node_step_m'] <= self.search_settings['morph_node_step_m']
                    and metrics['tangent_step_deg'] <= self.search_settings['morph_tangent_step_deg']
                    and metrics['curvature_component_step_per_m'] <= self.search_settings['morph_curvature_step_per_m'])
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                accepted = False
            if accepted:
                break
            increment *= .5
        if not accepted:
            self.capture_blend = before
        self.morph_history.append(dict(t=float(t),kind='contact_G_morph',accepted=accepted,
                                       before=before,after=self.capture_blend,metrics=metrics))

    def command(self,S):
        q,info=super().command(S)
        info.update(search_offset_applied_m=self.search_offset.copy(),
                    precontact_search_station_m=self.search_station.copy(),
                    contact_G_blend=self.capture_blend,
                    receiving_G_evidence_installed=self.contact_geometry_installed,
                    global_curve_dense=self.current_global_dense.copy())
        return q,info
