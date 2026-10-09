"""Extend the inner transfer material curve back to an all-source window."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from snake_control.inner_hook_route import InnerHookRoute
from snake_control.suspended_supply_transfer import LINK
from snake_control.suspended_transfer_route import R, BETA


class FullSourceInnerRoute(InnerHookRoute):
    def __init__(self, bridge=5):
        super().__init__(front=0, bridge=bridge)
        self.origin = self.source_cut
        self.source_start = 0
        self.total = 21*LINK-self.origin
        self.initial_residual = None
        self.frame_s = np.linspace(self.origin-21*LINK-.1, 21*LINK+.1, 6001)
        p = self.nominal(self.frame_s)
        # A real head rod follows the chord rather than the infinitesimal tangent.
        tangent = p-self.nominal(self.frame_s-LINK)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normals = []
        normal = np.array([0., p[0, 1], p[0, 2]-1.])
        normal -= normal@tangent[0]*tangent[0]
        normal /= np.linalg.norm(normal)
        for i, (s, direction) in enumerate(zip(self.frame_s, tangent)):
            if s <= self.source_cut:
                normal = np.array([0., p[i, 1], p[i, 2]-1.])
            elif i:
                cross = np.cross(tangent[i-1], direction)
                sine = np.linalg.norm(cross)
                if sine > 1e-12:
                    angle = np.arctan2(sine, np.clip(tangent[i-1]@direction, -1, 1))
                    normal = Rotation.from_rotvec(cross*angle/sine).apply(normal)
            normal -= normal@direction*direction
            normal /= np.linalg.norm(normal)
            normals.append(normal.copy())
        normals = np.asarray(normals)
        at = int(np.searchsorted(self.frame_s, LINK))
        desired = np.array([p[at, 0]-1.1, 0., p[at, 2]-1.])
        desired -= desired@tangent[at]*tangent[at]
        desired /= np.linalg.norm(desired)
        twist = np.arctan2(tangent[at]@np.cross(normals[at], desired), normals[at]@desired)
        for i, s in enumerate(self.frame_s):
            if s >= LINK:
                normal = np.array([p[i, 0]-1.1, 0., p[i, 2]-1.])
                normal -= normal@tangent[i]*tangent[i]
                normals[i] = normal/np.linalg.norm(normal)
            elif s > self.source_cut:
                a = np.clip((s-self.source_cut)/(LINK-self.source_cut), 0, 1)
                a = a*a*(3-2*a)
                normals[i] = Rotation.from_rotvec(tangent[i]*twist*a).apply(normals[i])
        self.tangents, self.normals = tangent, normals

    def frame(self, s):
        t = np.array([np.interp(s, self.frame_s, self.tangents[:, j]) for j in range(3)])
        t /= np.linalg.norm(t)
        n = np.array([np.interp(s, self.frame_s, self.normals[:, j]) for j in range(3)])
        n -= n@t*t
        n /= np.linalg.norm(n)
        return np.column_stack([n, np.cross(t, n), t])

    def bind_initial(self, points):
        self.initial = np.asarray(points).copy()
        v = self.origin-np.arange(22)*LINK
        self.initial_residual = CubicSpline(v[::-1], (self.initial-self.nominal(v))[::-1], axis=0)

    def window(self, shift, coeff=None):
        v = self.origin+shift-np.arange(22)*LINK
        p = self.nominal(v)
        a = self.blend(shift)
        if self.initial_residual is not None and a < 1:
            p += (1-a)*self.initial_residual(np.clip(v, self.origin-21*LINK, self.origin))
        if coeff is not None:
            w = np.clip((v-self.source_cut)/(-self.source_cut), 0, 1)
            h = 16*w*w*(1-w)*(1-w)
            p += a*h[:, None]*(coeff[0]+(2*w-1)[:, None]*coeff[1])
        return p, v

    def head_orientation(self, initial_rotation, shift, phase):
        return self.frame(self.origin+shift)@self.frame(self.origin).T@initial_rotation@Rotation.from_rotvec([0., 0., -phase]).as_matrix()

    def axial_progress_error(self, shift, centers, initial_centers):
        # No target-Y assumption while the head is still probing the bridge.
        expected = self.nominal([self.origin+shift])[0]-self.nominal([self.origin])[0]
        direction = self.frame(self.origin+shift)[:, 2]
        return float(direction@(expected-(centers[0]-initial_centers[0])))
