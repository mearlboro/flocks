#!/usr/bin/python3
import click
from enum import Enum
import numpy as np
import os

from analysis import plot
from flock.model import FlockModel
from flock.factory import *
from util.geometry import *
from util.util import *

from typing import Any, Callable, List, Dict, Tuple


class EnumParams(Enum):
    ALL               = 0
    VICSEK_ORDER      = 1
    MEAN_ANGLE        = 2
    STD_ANGLE         = 3
    CMASS             = 4
    MEAN_DIST_CMASS   = 5
    STD_DIST_CMASS    = 6
    MEAN_NEIGHBOURS   = 7
    MEAN_DIST_NEAREST = 8

    __titles__ = {
        'ALL'               : '',
        'VICSEK_ORDER'      : 'Vicsek order parameter',
        'MEAN_ANGLE'        : 'Mean player direction',
        'STD_ANGLE'         : 'Spread of player direction',
        'CMASS'             : 'Centre of mass',
        'MEAN_DIST_CMASS'   : 'Mean distance from centre',
        'STD_DIST_CMASS'    : 'Spread from centre',
        'MEAN_NEIGHBOURS'   : 'Mean number of interaction neighbours',
        'MEAN_DIST_NEAREST' : 'Mean distance to nearest neighbour'
    }

    __labels__ = {
        'ALL'               : '',
		'VICSEK_ORDER'      : '$v_a(t)$',
		'MEAN_ANGLE'        : '$\\tilde{\\theta}(t)$',
		'STD_ANGLE'         : '$\\sigma^2_{\\theta}(t)$',
		'CMASS'				: '$\\tilde{X}$',
		'MEAN_DIST_CMASS'   : '$\\tilde{d}_{\\tilde{X}}(t)$',
		'STD_DIST_CMASS'    : '$\\sigma_{\\tilde{d}}(t)$',
		'MEAN_NEIGHBOURS'   : '$\\tilde{\\rho}(t)$',
		'MEAN_DIST_NEAREST' : '$\\tilde{\\delta}(t)$'
	}

    def __str__(self) -> str:
        return self.name.lower()

    def title(self) -> str:
        return self.__titles__[self.name]

    def label(self) -> str:
        return self.__labels__[self.name]

    @classmethod
    def names(self) -> List[str]:
        return list(self.__members__.keys())

    @classmethod
    def members(self) -> List['EnumParams']:
        return list(self.__members__.values())



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
        vel = np.array([ [ ang_to_vec(a) * v / np.mean(V)
                           for a, v in zip(A,  V)  ]
                           for A, V in zip(At, Vt) ])
    else:
        vel = np.array([ [ ang_to_vec(a) * v for a in A ] for A in At ])

    avg_vel = np.mean(vel, axis = 1)
    avg_abs_vel = np.linalg.norm(avg_vel, 2, axis = 1) / v

    return avg_abs_vel


def __mean_std_angle(
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
    std_ang  = np.array([ np.sqrt(np.sum([ (am - a)**2  for a in A ]))
                          for (am, A) in zip(mean_ang, At) ]) / N

    return mean_ang, std_ang


def __mean_std_dist_cmass(
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

    cmass = [ centre_of_mass(X, L, bounds) for X in Xt ]
    dist  = np.array([ [ metric_distance(x, c, L, bounds)
                         for x in X ]
                         for (X, c) in zip(Xt, cmass) ])
    mean_dist = np.mean(dist, axis = 1)
    std_dist  = np.std( dist, axis = 1)

    return np.array(cmass), mean_dist, std_dist


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
    if r:
        ngh = np.array([ [ len(neighbours(i, X, r, EnumNeighbours.METRIC, bounds, L))
                        for i in range(len(X)) ]
                        for X in Xt ] )
        avg_ngh = np.mean(ngh, axis = 1)

        return avg_ngh
    else:
        return np.zeros(shape = len(Xt))


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


def param(
        param: 'EnumParams',
        Xt: np.ndarray,
        At: np.ndarray,
        L: int,
        r: float,
        bounds: EnumBounds,
        Vt: np.ndarray = None,
        v: float = 0.3,
    ) -> Dict['EnumParams', Any]:
    """
    Compute relevant order parameters given the trajectories of a system of self
    propelled particles. Implemented in this way to allow different order params
    to be computed in parallel for multiple systems if needed.

    Params
    ------
    param_name: str
        name of order parameter to be computed. If null, then return all order
        parameters as a dict
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
    dict with parameter name as key, and numpy array as value. If no `param_name`
    is given, then all order parameters are returned in the format below, other
    wise only the specified one is returned:

        EnumParams.VICSEK_ORDER:      (T,) Vicsek order parameter
        EnumParams.MEAN_ANGLE:        (T,) mean orientation
        EnumParams.STD_ANGLE:         (T,) std dev of orientation
        EnumParams.CMASS:             (T,D) coordinates of flock centre of mass
        EnumParams.MEAN_DIST_CMASS:   (T,) mean distance from centre of mass
        EnumParams.STD_DIST_CMASS:    (T,) std dev of distance from centre of mass
        EnumParams.MEAN_NEIGHBOURS:   (T,) mean number of interaction neighbours
        EnumParams.MEAN_DIST_NEAREST: (T,) mean distance to nearest neighbour
    """
    m = dict()
    import time

    if param == EnumParams.ALL:
        print('Computing Vicsek order parameter')
        start = time.time()
        m[EnumParams.VICSEK_ORDER] = __vicsek_order(At, Vt, v)
        print("Time elapsed: {}s".format(int(time.time() - start)))

        print('Computing mean & standard deviation of angle')
        start = time.time()
        m[EnumParams.MEAN_ANGLE], \
        m[EnumParams.STD_ANGLE] = __mean_std_angle(At)
        print("Time elapsed: {}s".format(int(time.time() - start)))

        print('Computing mean & standard deviation of distance from cmass')
        start = time.time()
        m[EnumParams.CMASS], \
        m[EnumParams.MEAN_DIST_CMASS], \
        m[EnumParams.STD_DIST_CMASS] = __mean_std_dist_cmass(Xt, bounds, L)
        print("Time elapsed: {}s".format(int(time.time() - start)))

        print('Computing mean number of neighbours')
        start = time.time()
        m[EnumParams.MEAN_NEIGHBOURS] = __mean_neighbours(Xt, r, bounds, L)
        print("Time elapsed: {}s".format(int(time.time() - start)))

        print('Computing mean distance to nearest neighbours')
        start = time.time()
        m[EnumParams.MEAN_DIST_NEAREST] = __mean_dist_nearest(Xt, bounds, L)
        print("Time elapsed: {}s".format(int(time.time() - start)))

    elif param == EnumParams.VICSEK_ORDER:
        print('Computing Vicsek order parameter')
        m[param] = __vicsek_order(At, Vt, v)

    elif param in [ EnumParams.MEAN_ANGLE, EnumParams.STD_ANGLE ]:
        print('Computing mean & standard deviation of angle')
        m[EnumParams.MEAN_ANGLE], \
        m[EnumParams.STD_ANGLE] = __mean_std_angle(At)

    elif param in [ EnumParams.MEAN_DIST_CMASS, EnumParams.STD_DIST_CMASS]:
        print('Computing mean & standard deviation of distance from cmass')
        m[EnumParams.CMASS], \
        m[EnumParams.MEAN_DIST_CMASS], \
        m[EnumParams.STD_DIST_CMASS] = __mean_std_dist_cmass(Xt, bounds, L)

    elif param == EnumParams.CMASS:
        print('Computing cmass')
        m[param] = np.array([ centre_of_mass(X, L, bounds) for X in Xt ])

    elif param == EnumParams.MEAN_NEIGHBOURS:
        print('Computing mean number of neighbours')
        m[EnumParams.MEAN_NEIGHBOURS] = __mean_neighbours(Xt, r, bounds, L)

    elif param == EnumParams.MEAN_DIST_NEAREST:
        print('Computing mean distance to nearest neighbours')
        m[EnumParams.MEAN_DIST_NEAREST] = __mean_dist_nearest(Xt, bounds, L)

    else:
        raise ValueError(f"Param {param_name} not supported")

    return m


@click.command()
@click.option('--path', required = True, help = 'Path to load model data from')
@click.option('--out',  required = True, help = 'Path to save data to', default = 'out/order/')
@click.option('--ordp', default = '', help = 'Order parameter to compute, all by default',
              type = click.Choice(EnumParams.names()))
@click.option('--redo', default = False,
              help = 'If data exists, recompute it, otherwise just redo plot')
def main(path: str, out: str, ordp: str, redo: bool) -> None:
    """
    After a simulation or experiment is run, compute (and plot) the results by
    showing trajectories, order parameters, susceptibilities, and histograms for the
    most 'interesting' states of that single run.

    It is assume that the simulation has a directory consistent with the mkdir
    method of the Flock abstract class, i.e. the dirname begins with the model
    name, followed by underscore, and other model details

        {name}(_{info})+(_{paramname}{paramval})+_{seed}?-id

    Run from the root pyflocks/ folder, will save the order parameters as CSV
    files in out/order and the plots in out/plt.

        python -m analysis.order [flags]
    """
    exp = FlockFactory.load(path)
    parampth = exp.mkdir(out)

    ords = dict()
    if ordp:
        ordp = EnumParams[ordp]
    else:
        ordp = EnumParams.ALL

    print(f"Computing order parameter(s) {ordp} for {path}, saving to {parampth}")

    if ordp != EnumParams.ALL:
        if os.path.exists(f"{parampth}/{ordp}.txt") and not redo:
            ords = { ordp: load_var(f"{parampth}/{ordp}.txt") }
        else:
            ords = param(ordp, exp.traj['X'], exp.traj['A'], exp.l, 0, exp.bounds)
            save_param(ords[ordp], str(ordp), parampth)
    else:
        for ordp in EnumParams.members()[1:]:
            if os.path.exists(f"{parampth}/{ordp}.txt") and not redo:
                ords[ordp] = load_var(f"{parampth}/{ordp}.txt")
            else:
                ords |= param(ordp, exp.traj['X'], exp.traj['A'], exp.l, 0, exp.bounds)
                save_param(ords[ordp], str(ordp), parampth)

    plot.order_params(exp, ords)
    # TODO: peak detection, plot those states
    plot.states([ 0, 50, 100, 150, 200, 300, 400, 499 ], exp)



if __name__ == "__main__":
    main()
