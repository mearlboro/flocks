#!/usr/bin/python
from enum import Enum
import numpy as np
from math import atan2, pi, fmod, sin, cos, sqrt

from typing import List, Tuple

class EnumBounds(Enum):
    """
    Specify whether particles bounce around the boundaries, or move to the other
    side in a toroidal world
    """
    PERIODIC   = 0
    REFLECTIVE = 1

class EnumNeighbours(Enum):
    """
    Specify whether a particle's neighbours are chosen to be in a radius r of
    the current particle, or the closest r neigbours
    """
    METRIC      = 0
    TOPOLOGICAL = 1


def vec_to_ang(v: np.ndarray) -> float:
    """
    Given a 2D vector return its angle using arctangent function

    Params
    ------
    v
        np.array of shape (1, 2) or list of floats. The first param to atan2
        must be the vertical component on the y-axis as per atan2 implementation
        to produce the consistent quadrant
        cf. https://en.wikipedia.org/wiki/Atan2

    Return
    ------
    angle as float number in interval [-pi, pi]
    """

    return atan2(v[1], v[0])


def ang_to_vec(theta: float) -> np.ndarray:
    """
    Given an angle returns a unit vector with origin in (0,0) at that angle.
    Module pi is applied, angle might be is greater than pi or smaller than -pi

    Params
    ------
    theta
        float number, should be in interval [-pi, pi]

    Returns
    ------
    numpy array of shape (1, 2) with the 2D coordinates of the vector's tip
    """

    if theta == -pi:
        theta = pi
    if theta > pi or theta < -pi:
        theta = fmod(theta, pi)

    x = np.array([0, 0])
    y = np.array([cos(theta), sin(theta)])

    return (y - x) / np.linalg.norm(y - x, 2)


def average_angles(A: np.ndarray) -> float:
    """
    Estimates average angle of all p angles in A taking into account possible
    negative angles should not cancel each-other out, following

        a = arctan2(sum(sin(A[i])), sum(cos(A[i])))

	cf. https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    Params
    -----
    A
        numpy array of shape (q, 1) storing the values of q angles in radians

    Returns
    ------
    average direction as a float
    """
    return np.arctan2(np.sum(np.sin(A)), np.sum(np.cos(A)))


def out_of_bounds(x: np.ndarray, L: int) -> bool:
    """
    Checks if 2D coordinates are out of bounds (0, 0) and (L, L)

    Params
    ------
    x
        numpy array of shape (2,) for coordinates
    L
        float size of the plae

    Returns
    ------
    True if out of bounds, False otherwise
    """
    return any(x < 0) or any(x > L)


def bounds_wrap(x: np.ndarray, L: int) -> np.ndarray:
    """
    Ensures point x is within bounds (0, 0) and (L, L) in a toroidal world

    Params
    ------
    x
        numpy array of shape (1, 2) for coordinates
    L
        size of the plane

    Returns
    ------
    numpy array of shape (1, 2)
    """
    if x[0] < 0:
      x[0] = L + x[0]

    if x[0] > L:
      x[0] = x[0] - L

    if x[1] < 0:
      x[1] = L + x[1]

    if x[1] > L:
      x[1] = x[1] - L

    return x


def bounds_reflect(
        x: np.ndarray, v: np.ndarray, dt: float, L: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures particle at coordinates in x is within bounds (0, 0) and (L, L) by
    adding specular reflections at the boundaries. The point (0, 0) is by
    convention bottom-right.

    When reaching the reflecting boundary, a particle will experience reflection
    and its velocity vector is updated according to

        V' = V - 2(v * Vn)Vn

    where Vn is the normal vector to the boundary

    For more details see Armbruster et al. (2017). "Swarming in bounded domains"
    Physica D: Nonlinear Phenomena. 344: 58-67.
    https://doi.org/10.1016/j.physd.2016.11.009

    Params
    ------
    x
        numpy array of shape (2,), 2D coordinates of point x at current time
    v
        numpy array of shape (2,), velocity vector of particle at current time
    dt
        time increment
    L
        size of the plane

    Returns
    ------
    updated velocity vector V'
    """

    if x[0] < 0:
        x[0] = -x[0]
        v[0] = -v[0]
    if x[0] > L:
        x[0] = 2*L - x[0]
        v[0] = -v[0]
    if x[1] < 0:
        x[1] = -x[1]
        v[1] = -v[1]
    if x[1] > L:
        x[1] = 2*L - x[1]
        v[1] = -v[1]

    return (x, v)


def neighbours(
        i: int,
        X: np.ndarray,
        r: float,
        topology: EnumNeighbours = EnumNeighbours.METRIC
    ) -> List[int]:
    """
    Get the neighbours of a given point of index i in the list of points X.
    Metric neighbours are within a given radius r and include i. Topological
    neighbours are the nearest r objects to point i.

    Params
    ------
    i
        particle/point to get neighbours for
    X
        numpy array of shape (N, D) with the coordinates of N points in
        a D-dimensional space
    r
        integer specifying radius for metric neighbours and the number
        of nearest points to retrieve for topological neighbours
    topology
        "metric" or "topological"

    Returns
    ------
    list of indexes of neighbouring particles, including current particle
    """

    N = len(X)

    if i >= N:
        raise ValueError("Index i must be smaller than number of particles N!")
    if r < 0:
        raise ValueError("Radius r must be positive")

    Xi = X[i, :]

    if topology == EnumNeighbours.METRIC:
        neighbours = [ j for j, Xj in enumerate(X) if np.linalg.norm(Xi - Xj, 2) < r ]
    else:
        if int(r) != r:
            raise ValueError("r must be an integer for topological neighbours")
        r = int(r)
        X_index = [ (j, X[j]) for j in range(0, N) ]
        X_index.sort(key = lambda jXj: np.linalg.norm(Xi - jXj[1], 2))
        neighbours = [ X_index[i][0] for i in range(0, r + 1) ]

    return neighbours


def centre_of_mass(X: np.ndarray, l: int = 0) -> np.ndarray:
    """
    Get coordinates of the centre of mass of all N points in the flock.
    If l is non-zero, normalize by the dimensions of the D-dimensional space.

    Params
    ------
    X
        numpy array of shape (N, D) with the coordinates of N points in
        a D-dimensional space of size l
    l
        size of space

    Returns
    ------
    numpy array of shape (D,) with the centre of mass coordinates
    """
    c = np.mean(X, axis = 0)
    if l:
        c /= l
    return c


def relative_positions(
        X: np.ndarray, c: np.ndarray, l: float = 0
    ) -> np.ndarray:
    """
    Compute N relative position vectors w.r.t. to a given point (can be centre
    of mass of the flock, or the center of the space in which the flock moves).
    If l is non-zero, normalize by the dimensions of the D-dimensional space.

    Params
    ------
    X
        numpy array of shape (N, D) with the coordinates of N points in
        a D-dimensional space of size l
    c
        numpy array of shape (D,) for the coordinates of the point of reference
    l
        size of space

    Returns
    ------
    numpy array of shape (N, D) with relative position vectors
    """
    R = np.array([ x - c for x in X ])
    if l:
        R /= l
    return R

