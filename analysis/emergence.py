"""
Calculate the quantities related to the PhiID theory of causal emergence, as
described in:

Rosas FE*, Mediano PAM*, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et
al. (2020) Reconciling emergences: An information-theoretic approach to
identify causal emergence in multivariate data. PLoS Comput Biol 16(12):
e1008289.

Uses the Java JIDT package for computing information-theoretic quantites, in
particular the MutualInfoCalculatorMultivariate.
See also https://github.com/jlizier/jidt/wiki/PythonExamples for using JIDT
in Python.
"""

import click
import jpype as jp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from typing import Callable, Dict, Iterable, List, NamedTuple, Tuple, Union

from analysis import order
from analysis.order import EnumParams as params
from flock.model import Flock, FlockModel
from flock.factory import FlockFactory
from util import util
from util.geometry import EnumBounds


class JVM:
    """
    Singleton class for managing the JVM for calls to infodynamics.jar
    """
    INFODYNAMICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'infodynamics.jar')

    @classmethod
    def start(self) -> None:
        """
        Start Java Virtual Machine to run Java code inside this Python repository
        """
        if not jp.isJVMStarted():
            print('Starting JVM...')
            try:
                jp.startJVM(jp.getDefaultJVMPath(), '-ea',
                            f"-Djava.class.path={self.INFODYNAMICS_PATH}")
                print('Done.')
            except:
                print('Error starting JVM')
                sys.exit(0)

    @classmethod
    def stop(self) -> None:
        """
        Gracefully exit Java Virtual Machine
        """
        if jp.isJVMStarted():
            print('Shutting down JVM...')
            jp.shutdownJVM()
            print('Done.')

    @classmethod
    def javify(self, X: np.ndarray) -> jp.JArray:
        """
        Convert a numpy array into a Java array to pass to the JIDT classes and
        functions.
        Given a 1-dim np array of shape (D,), return Java array[] of size D
        Given a 2-dim np array of shape (1, D), return Java array[] of size D
        Given a 2-dim np array of shape (D1, D2), return Java array[][] of size D1 x D2

        Params
        ------
        X
            numpy array of shape (D,) or (D1,D2) representing a time series

        Returns
        ------
        jX
            the X array cast to Java Array
        """
        X = np.array(X)

        if len(X.shape) == 1:
            dim = 1
            X = X[np.newaxis, :]
        else:
            dim = len(X.shape)

        if dim > 1:
            jX = jp.JArray(jp.JDouble, dim)(X.tolist())
        else:
            # special case to deal with scalars
            jX = jp.JArray(jp.JDouble, 1)(X.flatten())

        return jX



class Emergence(NamedTuple):
    """
    Tuple container for all results of emergence calculation, using named fields
    to make addressing each quantity and the decomposition of Psi easier.
    """
    psi: float
    psik1: float
    syn: float
    red: float
    corr: float
    gamma: float
    delta: float



def _MICalc(calcName: Callable[None, str]) -> Union[np.ndarray, float]:
    """
    Store the time series as observations in the mutual information calculators,
    used to estimate joint distributions between X and Y by using the pairs
    X[t], Y[t+dt], then compute the mutual information.

    Params
    ------
    X
        1st time series of shape (T, Dx) i.e. source.
        T is time or observation index, Dx is variable number.
    Y
        2nd time series of shape (T, Dy) i.e. target
        T is time or observation index, Dy is variable number.
    pointwise
        if set use pointwise MI rather than Shannon MI, i.e. applied on
        specific states rather than integrateing whole distributions.
        this returns a PMI value for each pair X[t], Y[t+dt] rather than
        a value for the whole time series
    dt
        source-destination lag

    Returns
    ------
    I(X[t], Y[t+dt]) for t in range(0, T - dt)
        if pointiwse = False, the MI is a float in nats not bits!
        if pointwise = True,  the MI is a list of floats, in nats, for every
                                pair of values in the two time series
    """
    def __compute(
            X: np.ndarray, Y: np.ndarray,
            pointwise: bool = False, dt: int = 0,
        ) -> np.ndarray:
        """
        Whenever Python decorator @_MICalc is used with a function, this function
        returns the mutual info as computed with the estimator specified by the
        function.
        """
        if len(X) != len(Y):
            raise ValueError('Cannot compute MI for time series of different lengths')

        jX, jY = (JVM.javify(X[dt:]), JVM.javify(Y[:-dt]))

        calc = jp.JClass(calcName())()
        calc.initialise(X.shape[1], Y.shape[1])
        #TODO: doesnt work
        #calc.setProperty('PROP_TIME_DIFF', str(dt))
        calc.setObservations(jX, jY)
        calc.finaliseAddObservations()

        if pointwise:
            # type JArray, e.g. <class 'jpype._jarray.double[]'>, can be indexed with arr[i]
            return calc.computeLocalUsingPreviousObservations(jX, jY)
        else:
            # float
            return calc.computeAverageLocalOfObservations()

    # Set name and docstrings of decorated function
    __compute.__name__ = calcName.__name__
    __compute.__doc__  = calcName.__doc__ + _MICalc.__doc__
    return __compute


class MutualInfo:
    """
    Class for calling various JIDT mutual information calculators for discrete
    and continuous variables. Returns class functions that can be passed to
    EmergenceCalc to compute mutual information.
    """
    @classmethod
    def get(self, name: str) -> Callable[None, str]:
        if name.lower() == 'gaussian':
            return self.ContinuousGaussian
        elif name.lower() == 'kraskov1':
            return self.ContinuousKraskov1
        elif name.lower() == 'kraskov2':
            return self.ContinuousKraskov2
        elif name.lower() == 'kernel':
            return self.ContinuousKernel
        else:
            raise ValueError(f"Estimator {name} not supported")


    @_MICalc
    def ContinuousGaussian() -> str:
        """
        Compute continuous mutual information (using differential entropy instead
        of Shannon entropy) between time series X and Y.
        The estimator assumes that the underlying distributions for the time
        series are Gaussian and that the time series are stationary.
        """
        return 'infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian'

    @_MICalc
    def ContinuousKraskov1() -> str:
        """
        Compute continuous mutual information using the Kraskov estimator between
        time series X and Y, specifically the 1nd algorithm in:

        Kraskov, A., Stoegbauer, H., Grassberger, P., "Estimating mutual information",
        Physical Review E 69, (2004) 066138.

        The mutual info calculator makes no Gaussian assumptions, but the time series
        must be stationary.
        """
        return 'infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1'

    @_MICalc
    def ContinuousKraskov2() -> str:
        """
        Compute continuous mutual information using the Kraskov estimator between
        time series X and Y, specifically the 2nd algorithm in:

        Kraskov, A., Stoegbauer, H., Grassberger, P., "Estimating mutual information",
        Physical Review E 69, (2004) 066138.

        The mutual info calculator makes no Gaussian assumptions, but the time series
        must be stationary.
        """
        return 'infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov2'

    @_MICalc
    def ContinuousKernel() -> str:
        """
        Compute continuous mutual information using box-kernel estimation between
        time series X and Y.
        The mutual info calculator makes no Gaussian assumptions, but the time series
        must be stationary.
        """
        return 'infodynamics.measures.continuous.kernel.MutualInfoCalculatorMultiVariateKernel'



class EmergenceCalc:
    """
    Computes quanities related to causal emergence using a given MutualInfo calculator
    function on time series X and V
    """
    def __init__(self,
            X: np.ndarray, V: np.ndarray,
            mutualInfo: Callable[[np.ndarray, np.ndarray, bool, int], float],
            pointwise: bool = False,
            dt: int =  1
        ) -> None:
        """
        Initialise class with the time series corresponding to the parts Xs and
        the whole V, and set any properties for the computation. By default, works
        with multivariate systems, so if any txn 2D array of t states of n scalar
        variable is passed, reshape the array to (t, n, 1) first.

        Params
        ------
        X
            system micro variables of shape (t, n, d) for n d-dimensional time
            series corresponding to the 'parts' in the system
        V
            candidate emergence feature: d-dimensional system macro variable of
            shape (t, d)
        mutualInfo
            mutual information function to use from MutualInfo class
        pointwise
            whether to use pointwise (p log p) or Shannon (sum p log p) MI
        dt
            number of time steps in the future to predict
        """
        print(f"Initialise Emergence Calculator using {'pointwise' if pointwise else 'Shannon'} mutual information with t'=t+{dt}")
        print(f"  and MI estimator {mutualInfo.__name__}")

        if len(X.shape) < 3:
            X = np.atleast_3d(X)
        if len(V.shape) < 2:
            V = V[:, np.newaxis]

        (t, n, d) = X.shape
        self.n = n
        self.V = V
        self.X = [ X[:, i] for i in range(n) ]
        print(f"  for {n} {d}-dimensional variables and {V.shape[1]}-dimensional macroscopic feature")
        print(f"  as time series of length {t}")

        print(f"Computing mutual informations: MI(V[t], V[t+{dt}])")
        self.vmiCalc = mutualInfo(V,  V,  pointwise, dt)

        print(f"Computing mutual informations: MI(Xi[t], V[t+{dt}])")
        self.xvmiCalcs = dict()
        for i in range(n):
            self.xvmiCalcs[i] = mutualInfo(self.X[i], V, pointwise, dt)

        print(f"Computing mutual informations: MI(V[t], Xi[t+{dt}])")
        self.vxmiCalcs = dict()
        for i in range(n):
            self.vxmiCalcs[i] = mutualInfo(V, self.X[i], pointwise, dt)

        print(f"Computing mutual informations: MI(Xi[t], Xj[t+{dt}])")
        self.xmiCalcs = dict()
        for i in range(n):
            for j in range(n):
                self.xmiCalcs[(i, j)] = mutualInfo(self.X[i], self.X[j], pointwise, dt)

        print('Done.')


    def psi(self,
            decomposition: bool = True, correction: int = 0
        ) -> Union[float, Tuple[float, float, float]]:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Psi = Synergy - Redundancy + Correction

        where:
            Synergy               MI(V(t); V(t'))
            Redundancy            sum_i MI(X_i(t); V(t'))
            1st order Correction  (N-1) min(MI(X_i(t); V(t'))

        where  t' - t = self.dt

        Params
        ------
        decomposition
            if True, return Synergy, Redundancy and Correction instead of Psi
        correction
            compute lattice correction of order given by this value. Currently
            supports only 0 and 1.

        Returns
        ------
        synergy
            mutual information between current and past values of V
        redundancy
            mutual information between past values of each micro variable X and
            current value of V
        correction
            the minimal mutual information between current and past values of X
            multiplied by N-1, to correct (if needed) the multiple counting of
            unique information in the redundancy term
        """
        msg = "Computing Psi "
        if correction:
            msg += f"using lattice correction of order {correction}"
        print(msg)

        syn  = self.vmiCalc
        red  = sum(xvmi for xvmi in self.xvmiCalcs.values())
        corr = 0

        if correction == 1:
            corr += (self.n - 1) * min(self.xvmiCalcs.values())

        if decomposition:
            return syn, red, corr
        else:
            return syn - red + corr


    def delta(self) -> float:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Delta = max_j (I(V(t);X_j(t')) - sum_i I(X_i(t); X_j(t'))

        where  t' - t = self.dt
        """
        delta = max(vx - sum(self.xmiCalcs[(i, j)] for i in range(self.n) if i != j)
                    for j, vx in enumerate(self.vxmiCalcs.values()) )
        return delta


    def gamma(self) -> float:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Gamma = max_j I(V(t); X_j(t'))

        where  t' - t = self.dt
        """
        gamma = max(self.vxmiCalcs.values())
        return gamma




def system(
        X: np.ndarray, V: np.ndarray, dts: List[int],
        mutualInfo: Callable, pointwise: bool = False, correction: int = 1
    ) -> List['Emergence']:
    """
    Initialise emergence calculator between time series X and V and return
    corrected, decomposed Psi, Gamma and Delta for time delays in dt.

    Assume stationarity, i.e. the underlying distribution for the value of
    variable i is the same regardless of time t. If the system is not stationary,
    use `ensemble` with `stationary = False` on multiple realisations of the
    system instead.

    Params
    ------
    Xs
        array of micro variables of shape (r, t, n, d1) for r realisations of a
        system of n d1-dimensional variables over t timesteps
    V
        array of emergence features of shape (r, t, d2) for each
    dts
        an array of all the numbers of time steps in the future to predict
    mutualInfo
        mutual information function to use from MutualInfo class
    pointwise
        whether to use pointwise (p log p) or Shannon (sum p log p) MI
    correction
        order of the lattice correction when computing Psi

    Returns
    ------
    decomposed Psi, Gamma, Delta as scalars in a (named) tuple foreach dt
    """
    emgs = []
    for dt in dts:
        calc = EmergenceCalc(X, V, mutualInfo, pointwise = pointwise, dt = dt)
        psi = calc.psi(decomposition = True, correction = correction)
        psi = ( psi[0] - psi[1], psi[0] - psi[1] + psi[2], *psi )
        emg = Emergence(*psi, calc.delta(), calc.gamma())
        emgs.append(emg)

    return emgs


def ensemble(
        stationary: bool,
        Xs: np.ndarray, Vs: np.ndarray, dts: int,
        mutualInfo: Callable, correction: int = 1,
        path: str = ''
    ) -> List[Tuple['Emergence', 'Emergence']]:
    """
    Compute emergence between time series of components X and a macroscopic
    feature V for a whole ensemble of realisations of the same system, for
    the time delays given in dts.

    When assuming stationarity, all Xi, V have the same probability distribution
    regardless of t, so the MI is applied between Xi and V at different times t
    and t', given by the each dt = t' - t.

    When the system is non-stationary, we cannot directly apply MI between Xi and
    V at arbitrary times (as the distribution of each Xi and V may be dependent
    on t). Instead, we take all Xi from the ensemble at the same time t, and V
    at the time t'.

    Params
    ------
    Xs
        numpy array of shape (r, t, n, d1) for ensembles of r realisations of
        a system of n d1-dimensional variables over t timesteps
    Vs
        numpy array of shape (r, t, d2) for the instantaneous d2-dimensional order
        parameter computed for each ensemble at each timestep
    dts
        array with a range of scalar time differences
    mutualInfo
        mutual information function to use from MutualInfo class
    correction
        order of the lattice correction when computing Psi
    path
        if set, dump the results of for the ensemble to path

    Returns
    ------
    a list of mean and standard deviation values for each emergence quantity as
    pairs of Emergence named tuples, with statistics for each dt
    """
    estats = []
    stats = lambda xs: (np.mean(xs, axis = 0), np.std(xs, axis = 0))

    R, T, N, D1 = Xs.shape[:4]
    if len(Vs.shape) > 2:
       D2 = Vs.shape[2]
    else:
       D2 = 1

    for dt in dts:
        print(f"Computing emergence quantities for a time delay of {dt}")
        emgs = []

        if stationary:
            # if we can assume stationarity, then we simply compute MI between
            # the time series with time delay dt
            for X, V in zip(Xs, Vs):
                emgs.append(system(X, V, [ dt ], mutualInfo)[0])
        else:
            # for non-stationary systems, we need to always compute MI between
            # all Xi across all R realisations at the same time t and t+dt, so
            # concatenate them and pass to the calculator with time delay T
            for t in range(1, T - dt - 1):
                print(f"Computing emergence from t={t} to t'={t*dt}")
                X  = np.concatenate((
                        Xs[:, t, :].reshape(R, N, D1),
                        Xs[:, t + dt, :].reshape(R, N, D1)))
                V  = np.concatenate((
                        Vs[:, t].reshape(R, D2),
                        Vs[:, t + dt].reshape(R, D2)))
                emgs.append(system(X, V, [ R ], mutualInfo)[0])

        mean, std = stats(np.array(emgs))
        estats.append((Emergence(*mean), Emergence(*std)))

    if path:
        np.save(f"{path}/ensemble_stats", estats)

    return estats



@click.command()
@click.option('--model',  help = 'Directory where system trajectories are stored')
@click.option('--est', type = click.Choice([ 'Gaussian', 'Kraskov1', 'Kraskov2', 'Kernel']),
              help = 'Mutual Info estimator to use', required = True)
@click.option('--decomposition', is_flag = True, default = False,
              help = 'If true, decompose Psi into the synergy, redundancy, and correction.')
@click.option('--correction', type = int, default = 1,
              help = 'Use nth-order lattice correction for emergence calculation.')
@click.option('--pointwise',  is_flag = True, default = False,
              help = 'If true, use pointwise mutual information for emergence calculation.')
@click.option('--threshold',
              help = 'Number of timesteps to wait before calculation, at least as many as the dimenstions of the system')
def test(model: str, est: str,
         decomposition: bool, correction: bool, pointwise: bool, threshold: int
    ) -> None:
    """
    Test the emergence calculator on the trajectories specified in `filename`, or
    on a random data stream.
    """

    if model:
        m = FlockFactory.load(model)
        X = m.traj['X']
        n = m.n
        M = order.param(params.CMASS, X, [], m.l, m.r, m.bounds)[params.CMASS]
    else:
        # generate data for 1000 timesteps for 2 binary variables
        np.random.seed(0)
        X = np.random.choice(a = [ False, True ], size = (1000, 2), p = [0.4, 0.6])
        # for the data to be emergent for dt=1, compute XOR of X with delay 1
        M = np.concatenate(([0], np.logical_xor(X[1:,0], X[1:,1])))

    est = MutualInfo.get(est)
    dts = [ 1, 2, 3  ]
    ems = system(X, M, dts, est, pointwise, correction)
    for dt, e in zip(dts, ems):
        print(f"{dt}: {e}")


if __name__ == "__main__":
    JVM.start()
    test()
    JVM.stop()
