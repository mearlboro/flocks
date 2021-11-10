#!/usr/bin/python
import math
import numpy as np
import os

from typing import Any, Dict, List


def save_var(X: np.ndarray, fn: str, path: str) -> None:
    """
    Append the variable `X` passed as param to the file `fn` in `path`.
    """
    n = len(X)

    with open(f'{path}/{fn}.txt', 'a') as f:
        for i in range(n):
            f.write( f'{X[i]}\t')

        f.write('\n')
    return


def load_var(filename: str) -> np.ndarray:
    """
    Load time series from file created with `save_var` and return as numpy array.

    Each file contains a variable for an N-dimensional system, with the index i
    on columns and the time t on rows. Values are floats separated by tabs.

    For example x can be the x-coordinate in 2D space of each particle or the
    angular velocity of each particle.

            x1  x2   ... xN
        t=1 .4  .7 . ... .3
        t=2 .1  ...  ...
        ...
        t=T ...

    The resulting array will have shape (T, N), with variable i at time t stored
    in X[t, i].
    """
    with open(filename, 'r') as f:
        X = [[x for x in line.split('\t') if x != '\n'] for line in f]

    return np.array(X).astype(float)
