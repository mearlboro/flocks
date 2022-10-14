import numpy as np

from analysis.emergence import EmergenceCalculator
from flock.model import FlockModel
from util.geometry import *

from typing import Any, List, Dict, Tuple


def __vicsek_order(
        At: np.ndarray, Vt: np.ndarray = None, v: float = 1.0
    ) -> np.ndarray:
    """
    Given the angles and absolute speeds of each particle in the system, return
    normalised absolute average velocity.

    The average absolute velocity is used as order parameter by Vicsek and is
    normalised by v. In a model with individual speeds each individual speed is
    used with no normalisation.

    Params
    ------
    At : numpy array of shape (T, N)
        angle of velocities for all the system variables across all time points
    Vt : numpy array of shape (T, N)
        absolute velocities for all the system variables across all time points,
        if the system uses different speeds for each particle
    v  : float
        absolute velocity if the system uses the same speed for each particle

    Returns
    ------
    numpy array of shape (T,) containing the Vicsek order parameter at each time
    step, valued 0 if all particles are moving chaotically and 1 if they align
    """
    if Vt is not None:
        vel = np.array([ [ ang_to_vec(a) * v
                           for a, v in zip(A,  V)  ]
                           for A, V in zip(At, Vt) ])
    else:
        vel = np.array([ [ ang_to_vec(a) * v for a in A ] for A in At ])

    avg_vel = np.mean(vel, axis = 1)
    avg_abs_vel = np.linalg.norm(avg_vel, 2, axis = 1) / v

    return avg_abs_vel


def __mean_var_angle(
        At: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the angles and absolute speeds of each particle in the system, return
    mean and spread from mean angle of direction of the flock. This is done
    using the circular mean and manually computing spreads from the mean.

    Params
    ------
    At : numpy array of shape (T, N)
        angle of velocities for all the system variables across all time points

    Returns
    ------
    Tuple of 2 numpy arrays
    - numpy array of shape (T,) containing the average direction
    - numpy array of shape (T,) containing spread from the average direction
    """
    (_, N) = At.shape

    mean_ang = np.array([ average_angles(A) for A in At ])
    var_ang  = np.array([ np.sum([ (am - a)**2
                          for a in A ])
                          for (am, A) in zip(mean_ang, At) ]) / N

    return mean_ang, var_ang


def __mean_var_dist_cmass(
        Xt: np.ndarray,
        bounds: EnumBounds,
        L: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the positions of all particles in the system find the std dev of
    distance to centre of mass

    Params
    ------
    Xt : numpy array of shape (T, N, 2)
        positions for all the system variables across all time points
    bounds
        periodic or reflective
    L
        size of space

    Returns
    ------
    Tuple of 3 numpy arrays
    - numpy array of shape (T, D) containing the coordinates of the centre
    - numpy array of shape (T,) containing the mean distance from the centre
    - numpy array of shape (T,) containing the spread from the centre
    """
    (T, N, D) = Xt.shape

    cmass = [ centre_of_mass(X, L, bounds)
              for X in Xt ]
    dist  = np.array([ [ metric_distance(x, c, L, bounds)
                         for x in X ]
                         for (X, c) in zip(Xt, cmass) ])
    mean_dist = np.mean(dist, axis = 1)
    var_dist  = np.var( dist, axis = 1)

    return cmass, mean_dist, var_dist


def __mean_neighbours(
        Xt: np.ndarray, r: float, bounds: EnumBounds, L: float = 0
    ) -> np.ndarray:
    """
    Given the positions of all particles in the system find the average number
    of interaction neighbours within a given radius.

    No need to apply this method on the topological interactions as always
    mean_neighbours(X) == r

    Params
    ------
    Xt : numpy array of shape (T, N, 2)
        positions for all the system variables across all time points
    r
        radius of metric interactions
    bounds
        periodic or reflective
    L
        size of space

    Returns
    ------
    numpy array of shape (T,) containing the average number of interaction
    neighbours in the system at each time-step
    """
    ngh = np.array([ [ len(neighbours(i, X, r, EnumNeighbours.METRIC, bounds, L))
                     for i in range(len(X)) ]
                     for X in Xt ] )
    avg_ngh = np.mean(ngh, axis = 1)

    return avg_ngh


def __mean_dist_nearest(
        Xt: np.ndarray, bounds: EnumBounds, L: float
    ) -> np.ndarray:
    """
    Given the positions of all particles in the system find the average minimum
    distance to the nearest neighbour.

    Params
    ------
    Xt : numpy array of shape (T, N, 2)
        positions for all the system variables across all time points
    bounds
        periodic or reflective
    L
        size of space for computing neighbours in periodic boundaries


    Returns
    ------
    numpy array of shape (T,) containing the mean distance to nearest neighbour
    in the system at each time-step
    """
    nears = np.array([ [ neighbours(i, X, 1, EnumNeighbours.TOPOLOGICAL, bounds, L)[-1]
                         for i in range(len(X)) ]
                         for X in Xt ])
    dists = np.array([ [ np.linalg.norm(x - X[n], 2)
                         for (x, n) in zip(X,  N) ]
                         for (X, N) in zip(Xt, nears) ])
    avg_dist = np.mean(dists, axis = 1)

    return avg_dist


def __psi_cmass(
        Xt: np.ndarray,
        cmass: np.ndarray
    ) -> Tuple[Dict[int, float], np.ndarray, np.ndarray]:
    """
    Given the positions of all particles in the system, compute the emergence
    Psi for the centre of mass of the system.

    Params
    ------
    Xt : numpy array of shape (T, N, D)
        positions for all the system variables across all time points
    cmass : numpy array of shape (N, D)
        positions for the centre of mass

    Returns
    ------
    A tuple of
    - a dictionary with the timestamps and highest and lowest Psi
    - a numpy arrays of shape (N,) containing Psi values with global observations
    - a numpy arrays of shape (N,) containing Psi values with local observations
    """
    calc_loc  = EmergenceCalculator(use_filter = False, use_local = True)
    calc_loc2 = EmergenceCalculator(use_filter = True,  use_local = True)
    calc_avg  = EmergenceCalculator(use_filter = False, use_local = False)

    (T, N, _) = Xt.shape

    loc  = []
    loc2 = []
    avg  = []
    for t in range(T):
        psi = calc_avg.update_and_compute(Xt[t], cmass[t])
        avg.append(psi)
        psi = calc_loc.update_and_compute(Xt[t], cmass[t])
        loc.append(psi)
        psi = calc_loc2.update_and_compute(Xt[t], cmass[t])
        loc2.append(psi)
    # exiting one calc gracefully closes the JVM for all of them
    calc_loc.exit()

    psi_dict = dict()
    psi_ind = [ (i, loc[i]) for i in range(0, T) ]
    psi_ind = psi_ind[::10]
    psi_ind.sort(key = lambda iPsi: iPsi[1], reverse = True)
    psi_dict.update({ i: Psi for i, Psi in psi_ind[:2] })
    psi_ind.sort(key = lambda iPsi: iPsi[1])
    psi_dict.update({ i: Psi for i, Psi in psi_ind[:2] })

    # the first N+buffer observations do not have a Psi computation
    thres = N if not calc_loc.use_filter else N + calc_loc.psi_buffer_size
    loc  = np.array(loc[N:])
    loc2 = np.array(loc2[N:])
    avg  = np.array(avg[N:])

    return psi_dict, loc, loc2, avg


def param(
        Xt: np.ndarray,
        At: np.ndarray,
        L: int,
        r: float,
        bounds: EnumBounds,
        Vt: np.ndarray = None,
        v: float = 1.0
    ) -> Dict[str, Any]:
    """
    Compute relevant order parameters given the trajectories of a system of self
    propelled particles

    Params
    ------
    Xt : numpy array of shape (T, N, 2)
        positions for all the system variables across all time points
    At : numpy array of shape (T, N)
        angle of velocities for all the system variables across all time points
    L
        size of space
    r
        radius of metric interactions or number of topological neighbours
    bounds
        whether the space's boundaries are reflective or periodic
    Vt : numpy array of shape (T, N)
        absolute velocities for all the system variables across all time points,
        if the system uses different speeds for each particle
    v  : float
        absolute velocity if the system uses the same speed for each particle

    Returns
    ------
    dict of numpy arrays, for each time step of the simulation, with the exception
    of the last item which is a dict
        'vicsek_order':      (T,) Vicsek order parameter
        'mean_angle':        (T,) mean orientation
        'std_angle':         (T,) std dev of orientation
        'cmass':             (T,D) coordinates of flock centre of mass
        'mean_dist_cmass':   (T,) mean distance from centre of mass
        'std_dist_cmass':    (T,) std dev of distance from centre of mass
        'mean_neighbours':   (T,) mean number of interaction neighbours
        'mean_dist_nearest': (T,) mean distance to nearest neighbour
        'psi_cmass_loc':     (T,) Psi of centre of mass computed with local MI, unfiltered
        'psi_cmass_loc2':    (T,) Psi of centre of mass computed with local MI, filtered
        'psi_cmass_avg':     (T,) Psi of centre of mass computed with average MI
        'psi_cmass_minmax:   (T,) Dict of int float pairs containing the time of highest
                                  unfiltered local Psi and the Psi value
    """
    m = dict()

    import time
    start = time.time()
    print('Computing Vicsek order parameter')
    m['vicsek_order'] = __vicsek_order(At, Vt, v)
    print("Time elapsed: {}s".format(int(time.time() - start)))

    print('Computing mean & standard deviation of angle')
    m['mean_angle'], m['var_angle'] = __mean_var_angle(At)
    print("Time elapsed: {}s".format(int(time.time() - start)))

    print('Computing mean & standard deviation of distance from cmass')
    m['cmass'], m['mean_dist_cmass'], m['var_dist_cmass'] = __mean_var_dist_cmass(Xt, bounds, L)
    print("Time elapsed: {}s".format(int(time.time() - start)))

    print('Computing mean number of neighbours')
    m['mean_neighbours']   = __mean_neighbours(Xt, r, bounds, L)
    print("Time elapsed: {}s".format(int(time.time() - start)))

    print('Computing mean distance to nearest neighbours')
    m['mean_dist_nearest'] = __mean_dist_nearest(Xt, bounds, L)
    print("Time elapsed: {}s".format(int(time.time() - start)))

    print('Computing Psi for cmass')
    m['psi_cmass_minmax'], m['psi_cmass_loc'], m['psi_cmass_loc2'], m['psi_cmass_avg'] = __psi_cmass(Xt, m['cmass'])
    print("Time elapsed: {}s".format(int(time.time() - start)))

    return m

