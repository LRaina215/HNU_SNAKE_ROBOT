"""Coordinate-free vector helpers. These do not contain scene geometry."""
import numpy as np


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('a finite nonzero direction is required')
    return vector / length


def orthogonal_frame(direction):
    direction = normalize(direction)
    # Keep the gauge stable around the task-side +Y prior. Choosing the least
    # aligned coordinate would jump between X and Z for arbitrarily small
    # direction changes near +Y.
    normal = np.array([1., 0., 0.])
    if abs(float(normal @ direction)) > .95:
        normal = np.array([0., 0., 1.])
    normal = normalize(normal - np.dot(normal, direction) * direction)
    return normal, np.cross(normal, direction)
