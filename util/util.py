#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from typing import Any, Dict, List


def sim_dir(root_dir: str, sim_dir: str) -> str:
    """
    Create a directory to store results of the current simulation. If it already
    exists, then add another one and return the path.

    Params
    -----
    root_dir { 'out/txt', 'out/img' }
        root directory for text dumps or plot exports
    sim_dir
        a directory for the current simulation, with a name of the form
            model_params-simID
    """

    pth = f'{root_dir}/{sim_dir}'

    while os.path.isdir(pth):
        # get the last sim ID
        if '-' in pth:
            [prefix, str_id] = pth.split('-')
            sim_id = int(str_id) + 1
            pth = f'{prefix}-{sim_id}'
        else:
            pth = f'{pth}-1'

    os.mkdir(pth)

    return pth


def dump_state(
        X: np.ndarray, filename: str, path: str
    ) -> None:
    """
    Dump the current state of the system variable passed as param to its
    corresponding file.

    Each system variable is dumped to an individual file correponding to that
    simulation.
    """
    n = len(X)

    with open(f'{path}/{filename}.txt', 'a') as f:
        for i in range(n):
            f.write( f'{X[i]}\t')

        f.write('\n')

    return


def load_var(filename: str) -> np.ndarray:
    """
    Load time series from file created with `dump_var` and return as numpy array.

    Each file contains a variable for an n-dimensional system, with the index i
    on columns and the time t on rows. Values are floats separated by tabs.

	A line represents a time step t. Position in D-dimensional space uses D
	columns, angular velocities and other scalars are one a single column

            x1  x2   ... xD
        t=1 .4  .7 . ... .3
        t=2 .1  ...  ...
        ...
        t=T ...

    The resulting array will have shape (T, D), with variable i at time t stored
    in X[t, i].
    """
    with open(filename, 'r') as f:
        X = [[x for x in line.split('\t') if x != '\n'] for line in f]

    return np.array(X).astype(float)


def position_matrix(Xs: List[np.ndarray]) -> np.ndarray:
    """
    Combine 2D numpy arrays representing individual coordinates in a
    D-dimensional space into 3D array

    Params
    ------
    Xs
        list of numpy arrays of shape (T, N)

    Returns
    ------
    numpy array of shape (T, N, D) for a D-dimensional system
    """
    (T, N) = Xs[0].shape

    return np.array([ np.stack([ X[i] for X in Xs ], axis = 1)
                      for i in range(T) ])
