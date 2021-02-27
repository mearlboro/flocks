#!/usr/bin/python3

import numpy as np
import numpy.random as rn
import os
from oct2py import octave, Oct2Py

from typing import Any, Dict, List, Tuple


def causal_emergence_criteria(X: np.ndarray, V: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute causal emergence criteria for a system with variables X with respect
    to macroscopic variable V

    Params
    ------
    X
        numpy array of shape (T, N*D), for system running for T time steps,
        with N D-dimensional variables
    V
        numpy array of shape (T,)

    Returns
    ------
    values of information theoretic quantities psi, delta, gamma as tuple

    Side-effects
    ------
    prints whether the system exhibits causal emergence, causal decoupling, or
    downward causation
    """
    psi   = oc.EmergencePsi(  X, V, 1, 'gaussian')
    delta = oc.EmergenceDelta(X, V, 1, 'gaussian')
    gamma = oc.EmergenceGamma(X, V, 1, 'gaussian')

    emergence = False

    if psi > 0:
        if gamma != 0:
            print("Causal emergence")
        else:
            print("Causal decoupling")

    if (delta > 0):
        print("Downward causation")

    return (psi, delta, gamma)


def example():
    # Random example
    print('Random')
    X = rn.randn(100,2)
    V = rn.randn(100,1)
    causal_emergence_criteria(X, V)

    # XOR example
    print('Xor')
    X = np.zeros([100, 2]).astype(int)
    for t in range(1,100):
        X[t, 0] = X[t-1, 0] ^ X[t-1, 1]
        X[t, 1] = rn.random() > 0.5
    V = X[:,0] ^ X[:,1]
    causal_emergence_criteria(X, V)


pth = f'{os.getcwd()}/ReconcilingEmergences'
octave.addpath(pth)
octave.savepath()
oc = Oct2Py()

example()
