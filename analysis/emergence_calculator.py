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

def emergence_psi(X, V, tau=1):
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
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=infodynamics.jar')
    calc_name = 'infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian'
    javify = lambda v: jp.JArray(jp.JDouble, 2)(v.tolist())

    ## Compute mutual info in macro variable
    vmiCalc = jp.JClass(calc_name)()
    vmiCalc.initialise(V.shape[1], V.shape[1])
    vmiCalc.setObservations(javify(V[:-tau]), javify(V[tau:]))
    psi = vmiCalc.computeAverageLocalOfObservations()

    ## Compute mutual info in every micro variable
    for Xi in X:
        xmiCalc = jp.JClass(calc_name)()
        xmiCalc.initialise(Xi.shape[1], V.shape[1])
        xmiCalc.setObservations(javify(Xi[:-tau]), javify(V[tau:]))
        psi -= xmiCalc.computeAverageLocalOfObservations()

    return psi


if __name__ == '__main__':

    X = [np.random.randn(1000, 2) for _ in range(10)]
    V = np.random.randn(1000, 2)

    print('Emergence for random data: %f'%emergence_psi(X, V))


