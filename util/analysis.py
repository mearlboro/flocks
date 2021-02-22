#!/usr/bin/python

import numpy as np
from scipy.stats import pearsonr

from util.geometry import ang_to_vec, vec_to_ang, centre_of_mass, relative_positions

from typing import Dict, List



def process_space(
        Xt: np.ndarray, l: float,
        relative_to: str = 'centre_of_mass', normalise: bool = False
    ) -> Dict[str, np.ndarray]:
    """
    Given the spatial coordinates of the system, run some basic analysis and
    return stats and spatial candidate macroscopic features (centre of mass etc)

    Params
    ------
    Xt
        all the spatial variables of the system across all time points as numpy
        array of shape (T, N, D), for T time steps, N variables in D dimensions
    l
        size of space
    relative_to: { 'centre_of_mass', 'centre_of_space' }
        choose whether relative vectors are computed with respect to centre of
        mass of the flock or the middle of the simulation space
    normalise
        if True, divide all values by the size of the space

    Returns
    ------
    dict of numpy arrays
      'rel_pos':        (T, N, D) the position vectors relative to centre
      'dist_cmass':     (T, N)    the distance (L2-norm) from centre
      'cmass':          (T, D)    flock's centre of mass
      'avg_dist_cmass': (T,)      the average distance relative to centre
      'var_dist_cmass': (T,)      the variance of distance relative to centre
    """
    if relative_to not in ( 'centre_of_mass', 'centre_of_space' ):
        raise ValueError("process_space: relative_to param must be \
            centre_of_mass or centre_of_space")

    (T, N, D) = Xt.shape

    cmass = [ centre_of_mass(X) for X in Xt ]

    if relative_to == 'centre_of_mass':
        rel = [ relative_positions(X, c) for (X, c) in zip(Xt, cmass) ]
    else:
        c = np.array([ l/2 ] * D)
        rel = [ relative_positions(X, c) for X in Xt ]

    m = dict()

    m['cmass']   = np.array(cmass)
    m['rel_pos'] = np.array(rel)

    dist = np.array([ [ np.linalg.norm(r, 2) for r in R ] for R in rel ])
    m['dist_cmass']     = dist
    m['avg_dist_cmass'] = np.mean(dist, axis = 1)
    m['var_dist_cmass'] = np.var( dist, axis = 1)

    return m


def process_angles(At: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Given the angles of each velocity vector in the system, return stats and
    related candidate macroscopic features
    (absolute average normalised velocity, average angle etc)

    Params
    ------
    At
        angle of velocities for all the system variables accross all time points
        as numpy array of shape (T, N)

    Returns
    ------
    dict of numpy arrays
        'avg_angle':   (T,) average angle
        'var_angle':   (T,) variance of angle
        'abs_avg_vel': (T,) absolute average velocity
        'var_abs_vel': (T,) variance of absolute average velocity
    """
    (T, N) = At.shape

    m = dict()

    vel     = np.array([ [ ang_to_vec(a) for a in A] for A in At ])
    avg_vel = np.mean(vel, axis = 1)
    var_vel = np.var( vel, axis = 1)

    m['avg_angle'] = np.array([ vec_to_ang(v) for v in avg_vel ])
    m['var_angle'] = np.array([ vec_to_ang(v) for v in var_vel ])

    m['avg_abs_vel'] = np.linalg.norm(avg_vel, 2, axis = 1)
    m['var_abs_vel'] = np.linalg.norm(var_vel, 2, axis = 1)

    return m



def autocorrelation(V: np.ndarray, window: int) -> np.ndarray:
    """
    Computes correlation between V[i] and V[i+w] for all w between 1 and window
    value

    Params
    ------
    window
        max window size to compute autocorrelation

    Returns
    ------
    np.array of shape (window,) with autocorrelation coefficients
    """

    (T,) = V.shape
    R = np.zeros((window,))

    for t in range(1, window + 1):
        V1 = V[:-t]
        V2 = V[t:]
        (R[t-1],_) = pearsonr(V1, V2)

    return R

