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
import pickle
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
    Class for calling various JIDT mutual information calculators for
    continuous variables. Returns class functions that can be passed
    to EmergenceCalc to compute mutual information.
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
            dt: int =  1,
            filename: str = ''
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

        if filename:
            print(f"Dumping EmergenceCalc object with all pairwise MI to {filename}_calc.pkl")
            with open(f"{filename}_calc.pkl", 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

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


    def gamma(self) -> float:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Gamma = max_j I(V(t); X_j(t'))

        where  t' - t = self.dt
        """
        gamma = max(self.vxmiCalcs.values())
        return gamma


    def delta(self) -> float:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Delta = max_j (I(V(t);X_j(t')) - sum_i I(X_i(t); X_j(t'))

        where  t' - t = self.dt
        """
        delta = max(vx - sum(self.xmiCalcs[(i, j)] for i in range(self.n))
                    for j, vx in enumerate(self.vxmiCalcs.values()) )
        return delta


class MutualInfos(NamedTuple):
    """
    Tuple container for all mutual info calculation, for each individual
    in the same system where necessary, before averaging the MI. Use
    named fields to make addressing each quantity easier.
    """
    vmi: float
    xvmi: List[float]
    vxmi: List[float]
    xiximi: List[float]
    xixjmi: List[float]

class MutualInfoStats(NamedTuple):
    """
    Tuple container for all results of average mutual info calculation,
    using named fields to make addressing each quantity easier.
    """
    vmi: float
    xvmi: float
    vxmi: float
    xiximi: float
    xixjmi: float

class EmergenceStats(NamedTuple):
    """
    Tuple container for all results of emergence calculation, using
    named fields to make addressing each quantity easier.
    """
    psik0: float
    psik1: float
    gamma: float
    delta: float



def system(
        X: np.ndarray, V: np.ndarray, dts: List[int],
        mutualInfo: Callable, pointwise: bool = False, path = ''
    ) -> List[Tuple['EmergenceStats', 'MutualInfos']]:
    """
    Initialise emergence calculator between time series X and V and return
    Psi, Gamma and Delta foreach delay in dts, as well as each relevant mutual
    information term in the causal theory of emergence.

    This function can be used regardless of the system being stationary or non-
    stationary, the X and V arrays should be constructed accordingly before
    being passed to `system`. See `ensemble` function for details.

    Params
    ------
    Xs
        array of micro variables of shape (t, n, d1) for the realisation
        of a system of n d1-dimensional variables over t timesteps, or
        for t realisations of an ensemble of n d1-dimensional variables
        at the exact same timestep
    V
        array of emergence features of shape (r, t, d2) for each
    dts
        list of all numbers of time steps in the future to predict
    mutualInfo
        mutual information function to use from MutualInfo class
    pointwise
        whether to use pointwise (p log p) or Shannon (sum p log p) MI
    path
        if set, save the data to path

    Returns
    ------
    list of tuples of EmergenceStats (4 floats) and MutualInfos (5 floats)
    we don't average MutualInfos into MutualInfoStats yet, as it may still be
    needed
    """
    results = []

    for dt in dts:
        calc = EmergenceCalc(X, V, mutualInfo, pointwise, dt = dt)

        e = EmergenceStats(
            psik0 = calc.psi(decomposition = False, correction = 0),
            psik1 = calc.psi(decomposition = False, correction = 1),
            gamma = calc.gamma(),
            delta = calc.delta(),
        )

        mi = MutualInfos(
            vmi    =   calc.vmiCalc,
            xvmi   = [ calc.xvmiCalcs[i]     for i in range(calc.n) ],
            vxmi   = [ calc.vxmiCalcs[i]     for i in range(calc.n) ],
            xiximi = [ calc.xmiCalcs[(i, i)] for i in range(calc.n) ],
            xixjmi = [ np.mean([ calc.xmiCalcs[(i, j)]
                                            for j in range(calc.n) if i != j ])
                                            for i in range(calc.n) ],
        )
        results.append((e, mi))

    if path:
        np.save(f"{path}/em_stats_mis", results)

    return results


def ensemble(
        stationary: bool,
        Xs: np.ndarray, Vs: np.ndarray, dts: int,
        mutualInfo: Callable, pointwise: bool = False,
        path: str = ''
    ) -> Tuple['EmergenceStats', 'MutualInfoStats']:
    """
    Compute emergence between time series of components X and a macroscopic
    feature V for a whole ensemble of realisations of the same system, for
    the time delays given in dts, and return an aveage value for each quantity
    for each dt, as well as the standard deviation across quantities for the Xi.

    When assuming stationarity, all Xi, V have the same probability distribution
    regardless of t, so the MI is applied between Xi and V at different times t
    and t', given by the each dt = t' - t.

    When the system is non-stationary, we cannot directly apply MI between Xi and
    V at arbitrary times (as the distribution of each Xi and V may be dependent
    on t). Instead, we take all Xi from the ensemble at the same time t, and V
    at the time t'. A large number of realisations is needed for a robust
    computation.

    Params
    ------
    stationary
        if set, assume system is stationary, intialise mutual info calc on the
        time series for each realisation. Otherwise, construct time series from
        realisations at the same timestep across the whole ensemble
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
    """
    emstats, mistats = [], []
    mean_std = lambda xs: (np.mean(xs, axis = 0), np.std(xs, axis = 0))

    R, T, N, D1 = Xs.shape[:4]
    if len(Vs.shape) > 2:
       D2 = Vs.shape[2]
    else:
       D2 = 1

    maxdt = max(dts)
    # most time we are interested in a given dt, or to observe evolution as dt
    # increases, therefore we may average over quantities for multiple systems
    # computed with the same dt
    for dt in dts:
        ems, mis = [], []
        if stationary:
            # if we can assume stationarity, then we simply compute MI between
            # the time series with time delay dt
            for X, V in zip(Xs, Vs):
                e, mi = system(X, V, [ dt ], mutualInfo, pointwise)[0]
                ems.append(e)
                mis.append(mi)
        else:
            # for non-stationary systems, we need to always compute MI between
            # all Xi across all R realisations at the same time t and t+dt, so
            # concatenate them and pass to the calculator with time delay R
            # we use the largest dt so we have the same number of calculations
            # to average over for all dts
            for t in range(1, T - maxdt - 1):
                print(f"Computing emergence from t={t} to t'={t+dt}")
                X  = np.concatenate((
                        Xs[:, t, :].reshape(R, N, D1),
                        Xs[:, t + dt, :].reshape(R, N, D1)))
                V  = np.concatenate((
                        Vs[:, t].reshape(R, D2),
                        Vs[:, t + dt].reshape(R, D2)))
                e, mi = system(X, V, [ R ], mutualInfo, pointwise)[0]
                ems.append(e)
                mis.append(mi)

        # since every emergence quantity is a scalar value, the mean and std are
        # accross ensembles if stationary, or over t if nonstationary
        emstats.append(mean_std(np.array(ems))) # has shape (2, 4)
        # for the individual MI over the mean and std should be computed over i
        # first get means over R or t-dt, the mean and std of those are over i
        if stationary:
            rt = R
        else:
            rt = T - maxdt - 2
        mis_stats = np.array([
           (np.mean([mis[t].vmi    for t in range(0, rt)]),
            np.std( [mis[t].vmi    for t in range(0, rt)])
           ),
            mean_std(np.mean([mis[t].xvmi   for t in range(0, rt)], axis = 0)),
            mean_std(np.mean([mis[t].vxmi   for t in range(0, rt)], axis = 0)),
            mean_std(np.mean([mis[t].xiximi for t in range(0, rt)], axis = 0)),
            mean_std(np.mean([mis[t].xixjmi for t in range(0, rt)], axis = 0))
            ]).T
        mistats.append(mis_stats) # has shape (2, 5)

    if path:
        stat = '_stat' if stationary else '_nonstat'
        np.save(f"{path}/ensemble_emstats{stat}_{min(dts)}-{max(dts)}", emstats)
        np.save(f"{path}/ensemble_mistats{stat}_{min(dts)}-{max(dts)}", mistats)

    return emstats, mistats


@click.command()
@click.option('--model', help = 'Directory where system trajectories are stored')
@click.option('--est', type = click.Choice([ 'Gaussian', 'Kraskov1', 'Kraskov2', 'Kernel']),
              help = 'Mutual Info estimator to use', required = True)
@click.option('--decomposition', is_flag = True, default = False,
              help = 'If true, decompose Psi into the synergy, redundancy, and correction.')
@click.option('--pointwise',  is_flag = True, default = False,
              help = 'If true, use pointwise mutual information for emergence calculation.')
@click.option('--threshold',
              help = 'Number of timesteps to wait before calculation, at least as many as the dimenstions of the system')
def test(model: str, est: str,
         decomposition: bool, pointwise: bool, threshold: int
    ) -> None:
    """
    Test the emergence calculator on the trajectories specified in `filename`, or
    on a random data stream.
    """
    pth = ''
    if model:
        m = FlockFactory.load(model)
        X = m.traj['X']
        n = m.n
        M = order.param(params.CMASS, X, [], m.l, m.r, m.bounds)[params.CMASS]
        pth = m.mkdir('out/order')
    else:
        # generate data for 1000 timesteps for 2 variables
        np.random.seed(0)
        X = np.random.normal(0, 1, size = (1000, 5))
        M = np.sum(X, axis = 1)

    est = MutualInfo.get(est)
    dts = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    results = system(X, M, dts, est, pointwise, pth)
    for dt, e in zip(dts, results):
        print(f"{dt}: {results[0]} {results[1]}")


if __name__ == "__main__":
    JVM.start()
    test()
    JVM.stop()
