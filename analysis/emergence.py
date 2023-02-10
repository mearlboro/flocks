"""
Calculate the quantities related to the PhiID theory of causal emergence, as
described in:

Rosas FE*, Mediano PAM*, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et
al. (2020) Reconciling emergences: An information-theoretic approach to
identify causal emergence in multivariate data. PLoS Comput Biol 16(12):
e1008289.
"""

import click
import jpype as jp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from typing import Callable, Dict, Iterable, List, Tuple, Union

from analysis import order
from analysis.order import EnumParams as params
from flock.model import Flock, FlockModel
from flock.factory import FlockFactory
from util import util
from util.geometry import EnumBounds


class JVM:

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



class MutualInfo:

    @classmethod
    def discrete(self,
        X: np.ndarray, Y: np.ndarray,
            pointwise: bool = False, dt: int = 0
        ) -> np.ndarray:
        # TODO: implement
        pass


    @classmethod
    def ContinuousGaussian(self,
            X: np.ndarray, Y: np.ndarray,
            pointwise: bool = False, dt: int = 0
        ) -> np.ndarray:
        """
        Compute continuous mutual information (using differential entropy instead
        of Shannon entropy) between time series X and Y.
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
            specific states rather than integrateing whole distributions
        dt
            source-destination lag

        Returns
        ------
        I(X[t], Y[t+dt]) for t in range(0, T - dt), in nats
        """
        if len(X) != len(Y):
            raise ValueError('Cannot compute MI for time series of different lengths')

        jX, jY = (JVM.javify(X[dt:]), JVM.javify(Y[:-dt]))

        calc = jp.JClass(
            'infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian')()
        #TODO: doesnt work
        #calc.setProperty('PROP_TIME_DIFF', str(dt))
        calc.initialise(X.shape[1], Y.shape[1])
        calc.setObservations(jX, jY)

        if pointwise:
            #TODO:
            #return calc.computeLocalUsingPreviousObservations()
            pass
        else:
            return calc.computeAverageLocalOfObservations()


class EmergenceCalc:

    def __init__(self,
            X: np.ndarray, V: np.ndarray,
            mutualInfo: Callable[[np.ndarray, np.ndarray, bool, int], float],
            pointwise: bool,
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
        X
            system micro variables of shape (N, T) for N components in the
            system
        V
            candidate emergence feature: D-dimensional system macro variable of
            shape (D, T)

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


    def delta(self
        ) -> float:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Delta = max_j (I(V(t);X_j(t')) - sum_i I(X_i(t); X_j(t'))

        where  t' - t = self.dt

        Params
        ------
        X
            system micro variables of shape (N, T) for N components in the
            system
        V
            candidate emergence feature: D-dimensional system macro variable of
            shape (D, T)

        Returns
        ------
        delta
        """
        j = 1
        delta = max(vx - sum(self.xmiCalcs[(i, j)] for i in range(self.n) if i != j)
                    for j, vx in enumerate(self.vxmiCalcs.values()) )
        return delta


    def gamma(self
        ) -> float:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Gamma = max_j I(V(t); X_j(t'))

        where  t' - t = self.dt

        Params
        ------
        X
            system micro variables of shape (N, T) for N components in the
            system
        V
            candidate emergence feature: D-dimensional system macro variable of
            shape (D, T)

        Returns
        ------
        gamma
        """
        gamma = max(self.vxmiCalcs.values())
        return gamma


@click.command()
@click.option('--model',  help = 'Directory where system trajectories are stored')
@click.option('--decomposition', is_flag = True, default = False,
              help = 'If true, decompose Psi into the synergy, redundancy, and correction.')
@click.option('--correction', type = int, default = 1,
              help = 'Use nth-order lattice correction for emergence calculation.')
@click.option('--pointwise',  is_flag = True, default = False,
              help = 'If true, use pointwise mutual information for emergence calculation.')
@click.option('--threshold',
              help = 'Number of timesteps to wait before calculation, at least as many as the dimenstions of the system')
def test(model: str,
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
        # generate data for 100 timesteps for 2 binary variables
        X = np.random.choice(a= [ False, True ], size = (100, 2), p = [0.5, 0.5])
        M = np.logical_xor(X[:,0], X[:,1])

    calc = EmergenceCalc(X, M, MutualInfo.ContinuousGaussian,
                            pointwise = pointwise, dt = 1)
    p = calc.psi(decomposition = decomposition, correction = correction)
    d = calc.delta()
    g = calc.gamma()

    print(p, d, g)


if __name__ == "__main__":
    JVM.start()
    test()
    JVM.stop()
