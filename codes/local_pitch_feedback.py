"""Pitch feedback from measured module centres, without pipe metadata.

The production cross-section construction joins link midpoints. A local arc
radius in that construction is not itself the wrapping radius. Fit the module
centres in the existing feedback-axis plane instead; retain the original circle
acceptance limits. Joint targets still use the production cross-section curve.
"""
import numpy as np

from local_pitch_geometry import estimate_local_radius


def estimate_module_radius(points_mm, axis, window=7):
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('invalid measured body points')
    centres = (points[:-1] + points[1:]) / 2
    modules = estimate_local_radius(centres, axis, window)
    nodes = []
    # Module centres have half-link arc-length coordinates. Interpolate their
    # radius field to the backbone's existing head/joint/tail knots. End values
    # are explicitly nearest-module extrapolations, not extra observations.
    for i in range(len(points)):
        indices = [0] if i == 0 else [len(modules)-1] if i == len(points)-1 else [i-1, i]
        source = [modules[j] for j in indices]
        valid = all(row['valid'] for row in source)
        nodes.append(dict(index=i, module_indices=indices, valid=valid,
                          radius_mm=float(np.mean([r['radius_mm'] for r in source]))
                          if all(r['radius_mm'] is not None for r in source) else None,
                          residual_mm=max(r['residual_mm'] for r in source)
                          if all(r['residual_mm'] is not None for r in source) else None,
                          condition=max(r['condition'] for r in source)
                          if all(r['condition'] is not None for r in source) else None,
                          coverage_deg=min(r['coverage_deg'] for r in source),
                          endpoint_extrapolated=len(indices)==1))
    return dict(modules=modules, nodes=nodes, centres_mm=centres.tolist(),
                estimand='local circle radius of measured link centres')
