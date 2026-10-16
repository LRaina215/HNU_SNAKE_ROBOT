"""Servo-only search reference and local-curvature command generation.

G, C and all displayed points are in an internal virtual chart. The chart has
no measured world pose, pipe endpoint, target center, or target radius. A
receiving helix is an explicit shape prior, not a geometrically identified pipe.
"""
from math import comb
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from native_mapping import NativeMapping, LINK
from geometry import normalize, orthogonal_frame


def reconstruct(field, bounds, frame, head, material_nodes):
    """Integrate the two material-curvature components into local curve C."""
    tangent, normal = frame[:, 0].copy(), frame[:, 1].copy()
    point = head.copy()
    curve = [point.copy()]
    for curvature, span in zip(field, np.diff(bounds) / 1000):
        binormal = np.cross(tangent, normal)
        omega = 1000 * (curvature[0] * binormal - curvature[1] * normal)
        rate = np.linalg.norm(omega)
        if rate > 1e-12:
            axis, angle = omega / rate, rate * span
            cosine, sine = np.cos(angle), np.sin(angle)
            point += span * (sine / angle * tangent + (1-cosine) / angle * np.cross(axis, tangent))
            tangent = tangent*cosine + np.cross(axis, tangent)*sine + axis*np.dot(axis, tangent)*(1-cosine)
            normal = normal*cosine + np.cross(axis, normal)*sine + axis*np.dot(axis, normal)*(1-cosine)
        else:
            point += span * tangent
        curve.append(point.copy())
    dense = np.asarray(curve)
    nodes = np.column_stack([np.interp(material_nodes, bounds, dense[:, j]) for j in range(3)])
    return dense, nodes


class ServoSearchCurve(NativeMapping):
    """Pure G -> curvature -> twenty-joint command mapping.

    The constants whitelist intentionally rejects location information by never
    reading it. x0, scene endpoints, actual body poses and target geometry do not
    affect this object. ``initial_command`` is the twenty commanded servo angles.

    Search offsets use virtual [x,y,z] meters. A two-vector is accepted as [x,z].
    ``receiving_direction`` is an optional servo-interaction direction candidate;
    it is not assumed to identify a cylinder axis. The controller owns its
    confidence, the rate limits and decisions to release/search/capture.
    """
    REQUIRED_CONSTANTS = ('radius', 'pitch', 'phi0', 'feed_per_cycle', 'feed_early',
                          'feed_late', 'tail_u0', 'anchor_u0',
                          'supply_anchor_normal', 'gauge_angle')

    def __init__(self, constants, initial_command):
        for key in self.REQUIRED_CONSTANTS:
            value = constants[key]
            setattr(self, key, np.asarray(value, float).copy() if isinstance(value, (list, tuple, np.ndarray)) else float(value))
        self.x0 = 0.0
        self.beta = 1 / np.sqrt(self.radius**2 + (self.pitch/(2*np.pi))**2)
        self.a = self.pitch * self.beta / (2*np.pi)
        self.supply_enabled = True
        self.release = self.roll_offset = self.capture_blend = 0.0
        self.search_azimuth = 0.0
        self.search_offset = np.zeros(3)
        self.local_relief = np.zeros(2)
        self.receiving_direction = np.array([0., 1., 0.])
        self.direction_from_servo_candidate = False
        self.initial_bias = None
        self._h = brentq(lambda h: (self.a*h)**2 + 4*self.radius**2*np.sin(self.beta*h/2)**2-LINK**2, LINK, 2*LINK)
        self.exit_u = -10*self._h
        self.nominal_search_station = np.array([3*LINK, 3*LINK, -.5*self.radius])
        command = np.asarray(initial_command, float)
        if command.shape != (20,) or not np.all(np.isfinite(command)):
            raise ValueError('initial_command must contain twenty finite servo angles')
        self._configure_initial(command)
        self.spec = dict(coordinate_system='virtual source chart; no world registration',
                         source_axis_gauge=[1, 0, 0], source_axis_origin=[0, 0, 0],
                         task_side_prior=[0, 1, 0], receiving_geometry='nominal helix prior, not a fitted cylinder',
                         allowed_inputs='nominal gait constants and commanded servo angles only',
                         local_action='first eight material intervals, two curvature components')

    def _source_jet(self, u, order=0):
        u = np.asarray(u)
        angle = self.phi0-self.beta*u
        if order == 0:
            return np.stack([self.a*(u+self.source_chart_shift), self.radius*np.cos(angle), self.radius*np.sin(angle)], axis=-1)
        if order == 1:
            return np.stack([np.full_like(u, self.a), self.radius*self.beta*np.sin(angle), -self.radius*self.beta*np.cos(angle)], axis=-1)
        return np.stack([np.zeros_like(u), -self.radius*self.beta**2*np.cos(angle), -self.radius*self.beta**2*np.sin(angle)], axis=-1)

    def _receiving_jet(self, u, order=0):
        """Open guide changing into an explicitly nominal receiving helix."""
        u = np.asarray(u)
        co,sn=np.cos(self.search_azimuth),np.sin(self.search_azimuth)
        rotation=np.array([[1.,0.,0.],[0.,co,-sn],[0.,sn,co]])
        axis = rotation@self.receiving_direction
        station = rotation@(self.nominal_search_station+self.search_offset)
        normal, binormal = orthogonal_frame(axis)
        phase = -np.pi/2 + self.beta*u
        radial = np.cos(phase)[..., None]*normal+np.sin(phase)[..., None]*binormal
        along = -np.sin(phase)[..., None]*normal+np.cos(phase)[..., None]*binormal
        radial0 = -binormal
        if order == 0:
            line = station+u[..., None]*axis
            # Both priors share the same entry point. No center is fitted.
            helix = station+self.a*u[..., None]*axis+self.radius*(radial-radial0)
        elif order == 1:
            line = np.broadcast_to(axis, u.shape+(3,))
            helix = self.a*axis+self.radius*self.beta*along
        else:
            line = np.zeros(u.shape+(3,))
            helix = -self.radius*self.beta**2*radial
        return (1-self.capture_blend)*line+self.capture_blend*helix

    def set_nominal_family(self, S):
        if not np.isfinite(S) or S < 0:
            raise ValueError('S must be finite and nonnegative')
        self.source_chart_shift = self.feed_early*self.phase_cells(S)
        self.spatial_phase = self.beta*self.source_chart_shift
        us = np.arange(-5., self.exit_u, .002)
        # At release zero the WHOLE available guide remains a source helix.
        # This permits source crawling before any release decision is made.
        v = np.linspace(0., 1., 241)
        up = self.exit_u*(1-v)
        u_tail = np.arange(.002, 4., .002)
        p0 = self._source_jet(self.exit_u)
        t0, a0 = self._source_jet(self.exit_u, 1), self._source_jet(self.exit_u, 2)
        end, t1, a1 = self._receiving_jet(0.), self._receiving_jet(0., 1), self._receiving_jet(0., 2)
        # A bounded outward polar connector preserves a source-side curved
        # support path instead of drawing a straight chord through the source.
        ph0 = self.phi0-self.beta*self.exit_u
        theta = 2*np.pi+np.arctan2(end[2], end[1])
        q0 = self.coords(p0, ph0)
        source_head = np.array([self.a*self.source_chart_shift, self.radius, self.phi0])
        q1 = (1-self.release)*source_head+self.release*self.coords(end, theta)
        dt0, dd0 = self.polar_derivatives(p0, t0, a0)
        dt1, dd1 = self.polar_derivatives(end, t1, a1)
        dt1 = (1-self.release)*np.array([self.a, 0., -self.beta])+self.release*dt1
        dd1 = self.release*dd1
        scale = (1-self.release)*(-self.exit_u)+self.release*3*LINK
        dt0, dt1, dd0, dd1 = dt0*scale, dt1*scale, dd0*scale**2, dd1*scale**2
        cp = np.zeros((8, 3))
        cp[0], cp[1], cp[2] = q0, q0+dt0/7, q0+2*dt0/7+dd0/42
        cp[7], cp[6], cp[5] = q1, q1-dt1/7, q1-2*dt1/7+dd1/42
        for j in (3, 4):
            desired = np.array([p0[0]+(j-2)*.08*LINK, (2.8 if j == 3 else 3.5)*LINK,
                                cp[2, 2]+(j-2)/3*(cp[5, 2]-cp[2, 2])])
            initial = np.array([self.a*(self.exit_u*(1-j/7)+self.source_chart_shift),
                                self.radius, ph0+(self.phi0-ph0)*j/7])
            cp[j] = (1-self.release)*initial+self.release*desired
        polar = sum(comb(7, j)*(1-v[:, None])**(7-j)*v[:, None]**j*cp[j] for j in range(8))
        prefix = np.column_stack([polar[:, 0], polar[:, 1]*np.cos(polar[:, 2]), polar[:, 1]*np.sin(polar[:, 2])])
        receiving_tail = self._receiving_jet(u_tail)
        receiving_polar = np.column_stack([receiving_tail[:, 0], np.linalg.norm(receiving_tail[:, 1:], axis=1),
                                           2*np.pi+np.unwrap(np.arctan2(receiving_tail[:, 2], receiving_tail[:, 1]))])
        source_polar = np.column_stack([self.a*(u_tail+self.source_chart_shift),
                                       np.full(len(u_tail), self.radius), self.phi0-self.beta*u_tail])
        tail_polar = (1-self.release)*source_polar+self.release*receiving_polar
        tail = np.column_stack([tail_polar[:, 0], tail_polar[:, 1]*np.cos(tail_polar[:, 2]),
                                tail_polar[:, 1]*np.sin(tail_polar[:, 2])])
        xyz = np.vstack([self._source_jet(us), prefix, tail])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        if np.min(np.diff(arc)) < 1e-10:
            raise ValueError('reference curve degenerates under the requested deformation')
        self.spline = CubicSpline(arc, xyz, axis=0)
        self.start_arc = float(arc[len(us)])
        self.target_arc = float(arc[len(us)+len(prefix)-1])
        self.connector_arc = self.target_arc-self.start_arc
        self.grow, self.recover = self.release, self.capture_blend
        self.source_exit = p0
        self.current_global_dense = self.spline(np.linspace(self.start_arc-.8, self.target_arc+.9, 257))

    def _configure_initial(self, initial_command):
        _, info = NativeMapping.command(self, 0.)
        bounds, nodes, field = info['curvature_bounds_mm'], info['material_arc_nodes_mm'], info['curvature_field']
        lengths = np.maximum(0., np.minimum(nodes[2:, None], bounds[None, 1:])-np.maximum(nodes[:-2, None], bounds[None, :-1]))
        bias = np.zeros_like(field)
        for component, mask in ((0, np.arange(20)%2 == 1), (1, np.arange(20)%2 == 0)):
            matrix = lengths[mask]
            residual = -initial_command[mask]-matrix@field[:, component]
            bias[:, component] = matrix.T@np.linalg.solve(matrix@matrix.T, residual)
        mid = .5*(bounds[:-1]+bounds[1:])
        self.initial_bias = (np.interp(mid, nodes, np.arange(22)), bias)

    def command(self, S, release=0., search_offset=(0., 0., 0.), roll_offset=0.,
                local_relief=(0., 0.), capture_blend=0., receiving_direction=None, search_azimuth=0.):
        offset = np.asarray(search_offset, float)
        if offset.shape == (2,):
            offset = np.array([offset[0], 0., offset[1]])
        relief = np.asarray(local_relief, float)
        if offset.shape != (3,) or relief.shape != (2,) or not np.all(np.isfinite(np.r_[offset, relief, S, release, roll_offset, capture_blend])):
            raise ValueError('nonfinite or incorrectly shaped curve command')
        if np.linalg.norm(offset) > 6*LINK:
            raise ValueError('search offset exceeds six-link virtual workspace bound')
        self.release = float(np.clip(release, 0., 1.))
        self.capture_blend = float(np.clip(capture_blend, 0., 1.))
        self.roll_offset = float(roll_offset)
        self.search_azimuth=float(search_azimuth)
        self.search_offset, self.local_relief = offset.copy(), relief.copy()
        self.receiving_direction = np.array([0., 1., 0.]) if receiving_direction is None else normalize(receiving_direction)
        self.direction_from_servo_candidate = receiving_direction is not None
        q, info = NativeMapping.command(self, float(S))
        global_points = info['reference_points'].copy()
        global_dense = info['nominal_curve_dense'].copy()
        native_field = info['curvature_field'].copy()
        field = native_field.copy()
        bounds, nodes = info['curvature_bounds_mm'], info['material_arc_nodes_mm']
        mid = .5*(bounds[:-1]+bounds[1:])
        preload = float(1-self.smooth(S/.75))
        if self.initial_bias is not None and preload > 0:
            coord = np.interp(mid, nodes, np.arange(22))
            old, bias = self.initial_bias
            field += preload*np.column_stack([np.interp(coord, old, bias[:, j]) for j in range(2)])
        v = np.clip(mid/nodes[8], 0., 1.)
        weight = np.sin(np.pi*v)**2
        weight[bounds[1:] > nodes[8]] = 0.
        field += weight[:, None]*relief[None, :]/1000
        bending = self.joint_bending(bounds, field, nodes)
        q = -np.where(np.arange(20)%2, bending[:, 0], bending[:, 1])
        dense, points = reconstruct(field, bounds, info['reference_frame'], global_points[0], nodes)
        info.update(curvature_field=field, native_curvature_field=native_field,
                    global_reference_points=global_points, global_reference_dense=global_dense,
                    global_curve_dense=self.current_global_dense.copy(),
                    nominal_curve_points=points, nominal_curve_dense=dense, reference_points=points,
                    local_relief_per_m=relief.copy(), local_relief_weight=weight,
                    initial_preload_fraction=preload, search_offset_m=offset.copy(),
                    virtual_search_station_m=(self.nominal_search_station+offset).copy(),
                    capture_blend=self.capture_blend, release=self.release,
                    search_azimuth_rad=self.search_azimuth,
                    receiving_direction=self.receiving_direction.copy(),
                    direction_from_servo_candidate=self.direction_from_servo_candidate,
                    coordinate_system='virtual_source_chart_no_world_registration',
                    receiving_geometry_identified=False,
                    local_curve_chord_error_mm=float(np.max(np.abs(np.linalg.norm(np.diff(points, axis=0), axis=1)-LINK)))*1000)
        if not np.all(np.isfinite(q)):
            raise ValueError('nonfinite curvature-generated joint command')
        return q, info
