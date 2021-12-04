"""
Simple function implementing the Psi emergence criterion related to
the PhiID theory of causal emergence, as described in:

Rosas FE*, Mediano PAM*, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et
al. (2020) Reconciling emergences: An information-theoretic approach to
identify causal emergence in multivariate data. PLoS Comput Biol 16(12):
e1008289.

This code depends on the JIDT function for the implementation of mutual
information. More information is available at:

    https://github.com/jlizier/jidt/wiki/UseInPython

Pedro Mediano, Oct 2021
"""
import numpy as np
import jpype as jp

from typing import Iterator

JIDT_PATH = 'infodynamics.jar'

def format(X: np.ndarray) -> Iterator[np.ndarray]:
    """
    Shape the trajectory array produced by a FlockingModel to an iterable of
    np.ndarray as required by the input to `emergence_psi`

    Params
    -------
    X
        np array containing with d-dimensional trajectories of a system with n
        parts across t timesteps - of shape (t, n, d)

    Returns
    ------
    an iterable of n numpy arrays of shape (t, d)
    """
    (_, n, _) = X.shape

    return iter([ np.array(X[:, i]) for i in range(n) ])


def emergence_psi(X: Iterator[np.ndarray], V: np.ndarray, tau: int = 1) -> float:
    """
    Calculate the emergence criterion Psi for a given set of micro variables X
    and a macro variable V, assuming they are jointly normally distributed (for
    the purposes of estimating mutual information).

    Parameters
    ----------
    X : iter of np.ndarray
        Iterable where each item contains a np.ndarray corresponding to each
        miscroscopic variable. Each np.ndarray must be of shape (T,Dx), where T
        is the number of timesteps and must be the same across variables, and
        Dx is the dimensionality, which may vary across variables.

    V : np.ndarray
        Array of shape (T,Dv) representing a supervenient feature of interest.

    tau : int
        Timescale for the emergence calculation, such that states of all
        variables at time t are used to predict states at time t+tau.


    Returns
    -------
    psi : float
        Emergence criterion. If psi > 0, we can conclude V is an emergent
        feature of X.
    """
    ## Some preliminaries for interfacing with Java
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', f'-Djava.class.path={JIDT_PATH}')
    calc_name = 'infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian'
    javify = lambda v, d: jp.JArray(jp.JDouble, d)(v.tolist())

    ## Compute mutual info in macro variable
    Dv = 1 if len(V.shape) == 1 else V.shape[1]
    vmiCalc = jp.JClass(calc_name)()
    vmiCalc.initialise(Dv, Dv)
    vmiCalc.setObservations(javify(V[:-tau], Dv), javify(V[tau:], Dv))
    whole = vmiCalc.computeAverageLocalOfObservations()


    ## Compute mutual info in every micro variable
    parts = 0
    for Xi in X:
        Dx = 1 if len(Xi.shape) == 1 else Xi.shape[1]
        xmiCalc = jp.JClass(calc_name)()
        xmiCalc.initialise(Dx, Dv)
        xmiCalc.setObservations(javify(Xi[:-tau], Dx), javify(V[tau:], Dv))
        parts += xmiCalc.computeAverageLocalOfObservations()

    return (whole - parts, whole, parts)


if __name__ == '__main__':

    X = [np.random.randn(1000, 2) for _ in range(10)]
    V = np.random.randn(1000, 2)

    print('Emergence for random data: %f'%emergence_psi(X, V))

