"""Clearance to the original compiled L-pipe collision geometry (metres).

Mesh halfspaces use actual compiled vertices in world coordinates. Outside a
convex mesh their maximum plane distance is a conservative distance bound;
inside they identify its nearest exit plane. Cylinders use their exact capped
cylinder distance. No model/physics parameters are changed.
"""
import mujoco
import numpy as np
from scipy.spatial import ConvexHull


class PipeClearance:
    def __init__(self, model, data):
        self.items = []
        for name in ['pipe_straight_01', 'pipe_sharp_in_02', 'pipe_sharp_out_03', 'pipe_straight_04']:
            g = model.geom(name).id
            position = data.geom_xpos[g].copy()
            rotation = data.geom_xmat[g].reshape(3, 3).copy()
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self.items.append(dict(name=name, kind='cylinder', position=position,
                    axis=rotation[:, 2], radius=float(model.geom_size[g, 0]), half=float(model.geom_size[g, 1])))
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                first, number = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[first:first+number]@rotation.T+position
                planes = np.unique(np.round(ConvexHull(vertices).equations, 12), axis=0)
                self.items.append(dict(name=name, kind='mesh', normals=planes[:, :3], offsets=planes[:, 3]))
            else:
                raise ValueError(f'Unsupported original pipe geometry: {name}')

    @staticmethod
    def distance(item, points):
        points = np.asarray(points).reshape(-1, 3)
        if item['kind'] == 'mesh':
            planes = points@item['normals'].T+item['offsets']
            face = np.argmax(planes, axis=1)
            return planes[np.arange(len(points)), face], item['normals'][face]
        relative = points-item['position']
        axial = relative@item['axis']
        radial = relative-axial[:, None]*item['axis']
        norm = np.linalg.norm(radial, axis=1)
        radial_normal = radial/np.maximum(norm[:, None], 1e-10)
        cap_normal = np.sign(axial)[:, None]*item['axis']
        q = np.column_stack([norm-item['radius'], abs(axial)-item['half']])
        outside = np.maximum(q, 0.)
        outside_length = np.linalg.norm(outside, axis=1)
        signed = outside_length+np.minimum(np.max(q, axis=1), 0.)
        gradient = np.where((q[:, 0] > q[:, 1])[:, None], radial_normal, cap_normal)
        out = outside_length > 0.
        gradient[out] = (outside[out, 0, None]*radial_normal[out]+outside[out, 1, None]*cap_normal[out])/outside_length[out, None]
        return signed, gradient

    def signed(self, points):
        shape = np.asarray(points).shape[:-1]
        distances = [self.distance(item, points)[0] for item in self.items]
        return np.min(distances, axis=0).reshape(shape)

    def project(self, points, margin):
        shape = np.asarray(points).shape
        point = np.asarray(points).reshape(-1, 3).copy()
        margin = np.broadcast_to(margin, shape[:-1]).reshape(-1)
        for _ in range(8):
            changed = False
            for item in self.items:
                distance, normal = self.distance(item, point)
                hit = distance < margin-1e-8
                if hit.any():
                    point[hit] += (margin[hit]-distance[hit]+1e-8)[:, None]*normal[hit]
                    changed = True
            if not changed:
                break
        return point.reshape(shape)

    def link_minimum(self, nodes, samples=25):
        fractions = np.linspace(0., 1., samples)
        lines = nodes[:-1, None, :]*(1-fractions[None, :, None])+nodes[1:, None, :]*fractions[None, :, None]
        return self.signed(lines).min(1)

    def guard_velocity(self, point, goal, velocity, radius=.029, supported_side=None, retained_margin=None):
        """Remove inward chase when its straight correction cuts a pipe."""
        velocity = velocity.copy()
        line = point+np.linspace(0., 1., 17)[:, None]*(goal-point)
        changed = False
        for item in self.items:
            current, normal = self.distance(item, point)
            path, _ = self.distance(item, line)
            side = 0 if item['name'] in ['pipe_straight_01', 'pipe_sharp_in_02'] else 1
            threshold = min(float(current[0])-.001, radius-.002)
            if retained_margin is not None and side == supported_side:
                threshold = max(.001, min(float(current[0])-.0015, retained_margin))
            if path.min() < threshold:
                inward = float(velocity@normal[0])
                if inward < 0.:
                    velocity -= inward*normal[0]
                    changed = True
        return velocity, changed
