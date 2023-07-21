#!/usr/bin/python
import math
import numpy as np
import re
import os

from typing import Any, Dict, List, Tuple

def proc_params(path: str) -> Tuple[List[str], int, Dict[str, float]]:
    """
    Given a model string, extract the parameters.

    For example from
        out/txt/Vicsek_periodic_metric_eta0.5_v0.1_r1_rho0.1_n10_999

    Get the following params dict

        { 'n': 10, 'e': 0.5, 'v': 0.1, 'r': 1, 'l': 10 }
    """
    # parse directory name to extract model parameters, exclude seed and ID
    d  = os.path.basename(path).split('-')[0]
    ps = d.split('_')
    seed = ps[-1]

    # some simulations may not have seed information
    try:
        seed = int(seed)
        ps = ps[:-1]
    except:
        seed = -1

    ps_dict = { re.findall('[a-z]+', p)[0]: float(re.findall('[0-9.]+', p)[0])
                for p in ps[3:]
                if len(re.findall('[0-9.]+', p)) }

    return ps, seed, ps_dict


def save_var(X: np.ndarray, fn: str, path: str) -> None:
    """
    Append the variable `X` passed as param to the file `fn` in `path`.
    """
    n = len(X)

    with open(f"{path}/{fn}.txt", 'a') as f:
        for i in range(n):
            f.write(f"{X[i]}\t")
        f.write('\n')
    return


def save_param(X: np.ndarray, fn: str, path: str) -> None:
    """
    Save the values in X, one per line
    """
    n = len(X)

    with open(f"{path}/{fn}.txt", 'a') as f:
        for i in range(n):
            if isinstance(X[i], (int, float)):
                f.write( f"{X[i]}\n")
            else:
                for x in X[i]:
                    f.write(f"{x}\t")
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

