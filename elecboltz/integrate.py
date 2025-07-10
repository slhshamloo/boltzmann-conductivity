import numpy as np
from typing import Callable, Sequence


def adaptive_octree_integrate(
        func: Callable, bounds: Sequence[float], depth: int = 7) -> float:
    """
    Integrate a binary-valued function over a cubic volume using an
    adaptive octree subdivision scheme.

    Parameters
    ----------
    func : Callable
        A function that takes three arguments ``(x, y, z)`` and returns
        a boolean value indicating whether the point is inside the
        volume of interest.
    bounds : Sequence[float]
        A collection of six floats
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``
        defining the bounds of the cubic volume to integrate over.
    depth : int, optional
        The maximum depth of the octree.

    Returns
    -------
    float
        The estimated integral of the function over the specified volume.
    """
    bounds = np.array([bounds]).transpose()
    integral = 0.0
    for _ in range(depth):
        corners, centers = _get_corners_and_centers(bounds)
        points = np.concatenate([corners, centers], axis=1)
        is_point_inside = func(points[0], points[1], points[2]
                               ).reshape((9, -1))

        is_voxel_inside = np.all(is_point_inside, axis=0)
        inside_bounds = bounds[:, is_voxel_inside]
        added_volume = np.sum(
            np.prod(inside_bounds[1::2] - inside_bounds[::2], axis=0))
        integral += added_volume
        
        is_voxel_partially_inside = np.logical_and(
            np.any(is_point_inside, axis=0), np.logical_not(is_voxel_inside))
        bounds = bounds[:, is_voxel_partially_inside]
        centers = centers[:, is_voxel_partially_inside]
        bounds = _subdivide_voxel(bounds, centers)
    integral += _integrate_final_octree(func, bounds)
    return integral


def _get_corners_and_centers(bounds):
    corners = np.concatenate(
        [np.vstack([bounds[i], bounds[j], bounds[k]])
         for i in range(2) for j in range(2, 4) for k in range(4, 6)],
        axis=1)
    centers = (bounds[::2] + bounds[1::2]) / 2
    return corners, centers

def _subdivide_voxel(bounds, centers):
    return np.concatenate([
        np.vstack([bounds[0], centers[0], bounds[2], centers[1],
                   bounds[4], centers[2]]),
        np.vstack([centers[0], bounds[1], bounds[2], centers[1],
                   bounds[4], centers[2]]),
        np.vstack([bounds[0], centers[0], centers[1], bounds[3],
                   bounds[4], centers[2]]),
        np.vstack([centers[0], bounds[1], centers[1], bounds[3],
                   bounds[4], centers[2]]),
        np.vstack([bounds[0], centers[0], bounds[2], centers[1],
                   centers[2], bounds[5]]),
        np.vstack([centers[0], bounds[1], bounds[2], centers[1],
                   centers[2], bounds[5]]),
        np.vstack([bounds[0], centers[0], centers[1], bounds[3],
                   centers[2], bounds[5]]),
        np.vstack([centers[0], bounds[1], centers[1], bounds[3],
                   centers[2], bounds[5]])],
        axis=1)


def _integrate_final_octree(func, bounds):
    corners, centers = _get_corners_and_centers(bounds)
    volumes = np.prod(bounds[1::2] - bounds[::2], axis=0)
    return np.sum((np.sum(func(corners[0], corners[1], corners[2]
                               ).reshape((8, -1)), axis=0)
                   + func(centers[0], centers[1], centers[2])) / 9
                  * volumes)
