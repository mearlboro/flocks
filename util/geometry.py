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


def ang_mod(a: float) -> float:
    """
    Given an angle in radians, if it is larger than pi or smaller than -pi
    subtract or add the required rotations around the circle

    Params
    ------
    a
        float representing angle in radians

    Returns
    ------
    float representing angle in radians in interval (-pi, pi]
    """
    if a == -pi:
        a = pi
    if a > pi or a < -pi:
        a = fmod(a, pi)
    return a


def vec_to_ang(v: np.ndarray) -> float:
    """
    Given a 2D vector return its angle using arctangent function

    Params
    ------
    v
        np.array of shape (1, 2) or list of floats. The first param to atan2
        must be the vertical component on the y-axis to produce the consistent
        quadrant
        cf. https://en.wikipedia.org/wiki/Atan2

    Return
    ------
    angle as float number in interval [-pi, pi]
    """

    return atan2(v[1], v[0])


def ang_to_vec(a: float) -> np.ndarray:
    """
    Given an angle returns a unit vector with origin in (0,0) at that angle.
    Module pi is applied if the angle is greater than pi or smaller than -pi

    Params
    ------
    a
        float number, should be in interval [-pi, pi]

    Returns
    ------
    numpy array of shape (2,) with the 2D coordinates of the vector's tip
    """

    a = ang_mod(a)
    x = np.array([cos(a), sin(a)])

    return x / np.linalg.norm(x, 2)


def bearing_to(a: float, x: np.ndarray) -> float:
    """
    Compute the angular difference between an angle a at the origin and the angle
    formed by a vector from origin to the point x.

    This angle is referred to as 'bearing' and they are usually measured in a
    clockwise direction.

    Params
    ------
    a
        float in range (-pi, pi] representing angle in radians
    x
        np.array of size (D,) for a point in D-dimensional space

    Returns
    ------
    angle in radians
    """
    ax = atan2(x[1], x[0])
    d  = ang_mod(ax - a + pi) - pi

    return d


def average_angles(A: np.ndarray) -> float:
    """
    Estimates average angle of all q angles in A taking into account possible
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
    return atan2(sum(np.sin(A)), sum(np.cos(A)))


def sum_vec_ang(A: List[float], V: List[float]) -> np.ndarray:
    """
    Sum given vectors with angle stored in A and absolute velocities in V

    Params
    ------
    A
        list of floats representing angles in interval [-pi, pi]
    V
        list of floats representing absolute velocities

    Returns
    ------
    numpy array of shape (2,) with the 2D coordinates of the sum vector's tip
    """
    if (len(A) != len(V)):
        raise ValueError("Each angle must have a corresponding speed")
        exit(0)

    vecs = np.array([ ang_to_vec(a) * v for a,v in zip(A, V) ])
    sumv = np.sum(vecs, axis = 0)

    return sumv


def out_of_bounds(x: np.ndarray, L: float) -> bool:
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


def bounds_wrap(x: np.ndarray, L: float) -> np.ndarray:
    """
    Ensures point x is within bounds (0, 0) and (L, L) in a toroidal world

    Params
    ------
    x
        numpy array of shape (2,) for coordinates
    L
        size of the plane

    Returns
    ------
    numpy array of shape (2,)
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
        x: np.ndarray, v: np.ndarray, L: float
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
        numpy array of shape (2,) for 2D coordinates of point x at current time
    v
        numpy array of shape (2,), velocity vector of particle at current time
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
    if r <= 0:
        raise ValueError("Radius r must be strictly positive")

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


def periodic_mean(x: np.ndarray, L: float) -> float:
    """
    Given an array of 1-dimensional coordinates and the size of the space
    it computes mean by taking into account the boundaries by mapping the
    data to points on the unit circle
    Cf. Bai, Breen (2008). "Calculating Center of Mass in an Unbounded 2D
    Environment". Journal of Graphics Tools, vol. 13 - Issue 4, pp 53-60.
    https://doi.org/10.1080/2151237X.2008.10129266

    Params
    ------
    x
        numpy array of shape (N,) with one dimension of coordinates of N points
    L
        the size of the space in that dimension

    Returns
    ------
    centre of mass of all coordinates accounting for the periodic boundaries
    """
    cmap  = x / L * 2 * pi - pi
    cmean = atan2(np.sin(cmap).mean(), np.cos(cmap).mean())
    mean  = (cmean + pi) / (2 * pi) * L

    return mean


def periodic_diff(
        x: np.ndarray, y: np.ndarray, L: float, norm: bool = False
    ) -> np.ndarray:
    """
    Given two points in a D-dimensional space of size LxL with periodic bounds,
    return the difference vector between them using the minimum image convention
    cf. https://en.wikipedia.org/wiki/Periodic_boundary_conditions

    Params
    -----
    x
        numpy array of shape (D,) with the coordinates of the first point
    y
        numpy array of shape (D,) with the coordinates of the second point
    L
        size of space
    norm
        if set, return the metric distance (L2 norm) between the two vectors,
        otherwise return the difference vector
    """
    imgs = [ [0, 0], [0, L], [0, -L], [L, 0], [L, L], [L, -L], [-L, 0], [-L, L], [-L, -L] ]
    diff = [ (x + i - y, np.linalg.norm(x + i - y)) for i in imgs ]
    res  = min(diff, key = lambda d: d[1])

    return res[1] if norm else res[0]


def centre_of_mass(
        X: np.ndarray, L: float, topology: EnumBounds
    ) -> np.ndarray:
    """
    Get coordinates of the centre of mass of all N points in the flock.
    If L is non-zero, normalize by the dimensions of the D-dimensional space.

    Params
    ------
    X
        numpy array of shape (N, D) with the coordinates of N points in
        a D-dimensional space of size l
    L
        size of space
    topology
        whether the space's boundaries are reflective or periodic

    Returns
    ------
    numpy array of shape (D,) with the centre of mass coordinates
    """
    (_, D) = X.shape

    if topology == EnumBounds.PERIODIC:
        c = np.array([ periodic_mean(X[:, d], L) for d in range(D) ])
    else:
        c = np.mean(X, axis = 0)

    return c


def relative_positions(
        X: np.ndarray, c: np.ndarray, L: float, topology: EnumBounds
    ) -> np.ndarray:
    """
    Compute N relative position vectors w.r.t. to a given point (can be centre
    of mass of the flock, or the center of the space in which the flock moves).
    We use the minimum image convention for computing algorithms for periodic
    boundaries cf. https://en.wikipedia.org/wiki/Periodic_boundary_conditions

    Params
    ------
    X
        numpy array of shape (N, D) with the coordinates of N points in
        a D-dimensional space of size l
    c
        numpy array of shape (D,) for the coordinates of the point of reference
    L
        size of space
    topology
        whether the space's boundaries are reflective or periodic

    Returns
    ------
    numpy array of shape (N, D) with relative position vectors
    """
    if topology == EnumBounds.PERIODIC:
        R = np.array([ periodic_diff(x, c, L) for x in X ])
    else:
        R = np.array([ x - c for x in X ])

    return R

