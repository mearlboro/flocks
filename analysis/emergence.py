"""
Simple class implementing a running calculator for the quantities related to
the PhiID theory of causal emergence, as described in:

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

from typing import Callable, Iterable

from flock.model import FlockModel
from util.geometry import centre_of_mass
from util import util


INFODYNAMICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'infodynamics.jar')
PSI_START = -5

def javify(Xi: np.ndarray) -> jp.JArray:
    """
    Convert a numpy array into a Java array to pass to the JIDT classes and
    functions.
    Given a 1-dimensional np array of shape (D,)  , return Java array of size D
    Given a 2-dimensional np array of shape (1, D), return Java array of size D

    Params
    ------
    Xi
        numpy array of shape (D,) or (1,D) representing one 'micro' part of the
        system or one value of the macroscopic feature

    Returns
    ------
    jXi
        the Xi array cast to Java Array
    """
    if len(Xi.shape) == 1:
        D = Xi.shape[0]
        Xi = Xi[np.newaxis, :]
        jXi = jp.JArray(jp.JDouble, D)(Xi.tolist())
    else:
        D = Xi.shape[1]
        jXi = jp.JArray(jp.JDouble, D)(Xi.tolist())

    return jXi


class EmergenceCalculator():
    def __init__(self,
            use_correction: bool = True,
            use_filter: bool = True,
            psi_buffer_size : int = 20,
            observation_window_size : int = -1,
            use_local : bool = True,
            sample_threshold: int = 0
        ) -> None:
        """
        Construct the emergence calculator by setting member variables and
        checking the JVM is started. The JIDT calculators will be initialised
        later, when the first batch of data is provided.

        After calculating the value of emergence for a given frame, it is
        median-filtered with recent past values to reduce volatility.

        Parameters
        ----------
        use_correction : bool
            Whether to use the 1st-order lattice correction for emergence
            calculation. (default: true)
        use_filter : bool
            Whether to use the median filter (default: true).
        psi_buffer_size : int
            Number of past emergence values used for the median filter
            (default: 20).
        observation_window_size : int
            Number of past observations to take into account for the calculation
            of psi. If negative or zero, use all past data (default: -1).
        use_local : bool
            If true, computes psi the local (i.e. pointwise) mutual info of
            the latest sample. If false, uses the standard (i.e. average) mutual
            info of the observation window (default: true).
        sample_threshold : int
            Number of timesteps to wait before calculation, at least as many as
            the dimenstions of the system. If smaller, will automatically use a
            value equal to the number of the variables in the system. (default: 0)
        """

        self.is_initialised = False
        self.sample_counter = 0

        self.use_filter = use_filter
        self.use_correction = use_correction
        self.psi_buffer_size = psi_buffer_size
        self.sample_threshold = sample_threshold
        self.past_psi_vals = [PSI_START] * psi_buffer_size

        self.observation_window_size = observation_window_size
        self.observations_V = []
        self.observations_X = []

        self.use_local = use_local

        if not jp.isJVMStarted():
            jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s'%INFODYNAMICS_PATH)

        print(f'Successfully initialised EmergenceCalculator.')
        print(f' observations:   {observation_window_size},')
        print(f' use correction: {use_correction}')
        print(f' use local:      {use_local}')
        print(f' use filter:     {use_filter}')
        if use_filter:
            print(f' buffer: {psi_buffer_size}')


    def initialise_calculators(self, X: np.ndarray, V: np.ndarray) -> None:
        """
        """
        self.N = len(X)
        if self.sample_threshold < self.N:
            self.sample_threshold = self.N

        V = V[np.newaxis, :]
        self.xmiCalcs = []
        for Xi in X:
            Xi = Xi[np.newaxis, :]
            self.xmiCalcs.append(jp.JClass('infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian')())
            self.xmiCalcs[-1].initialise(Xi.shape[1], V.shape[1])
            self.xmiCalcs[-1].startAddObservations()

        self.vmiCalc = jp.JClass('infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian')()
        self.vmiCalc.initialise(V.shape[1], V.shape[1])
        self.vmiCalc.startAddObservations()

        self.is_initialised = True


    def update_calculators(self, V: np.ndarray) -> None:
        """
        """
        jV = javify(V)
        jVp = javify(self.past_V)
        jXp = [javify(Xip) for Xip in self.past_X]

        if self.observation_window_size <= 0:
            self.vmiCalc.addObservations(jVp, jV)
            for jXip,calc in zip(jXp, self.xmiCalcs):
                calc.addObservations(jXip, jV)

        else:
            self.observations_V.append((jVp, jV))
            self.observations_X.append((jXp, jV))
            if len(self.observations_V) > self.observation_window_size:
                self.observations_V.pop(0)
                self.observations_X.pop(0)

            self.initialise_calculators(self.past_X, V)
            for jVp, jV in self.observations_V:
                self.vmiCalc.addObservations(jVp, jV)
            for jXp, jV in self.observations_X:
                for jXip,calc in zip(jXp, self.xmiCalcs):
                    calc.addObservations(jXip, jV)


    def compute_psi(self, V: np.ndarray) -> float:
        """
        """
        self.vmiCalc.finaliseAddObservations()
        jV = javify(V)

        if self.use_local:
            psi = self.vmiCalc.computeLocalUsingPreviousObservations(
                    javify(self.past_V), jV)[0]
            for Xip,calc in zip(self.past_X, self.xmiCalcs):
                calc.finaliseAddObservations()
                psi -= calc.computeLocalUsingPreviousObservations(javify(Xip), jV)[0]

            if self.use_correction:
                marginal_mi = [ calc.computeAverageLocalOfObservations()
                                for calc in self.xmiCalcs ]
                psi += (self.N - 1) * np.min(marginal_mi)

        else:
            psi = self.vmiCalc.computeAverageLocalOfObservations()
            for calc in self.xmiCalcs:
                calc.finaliseAddObservations()
                psi -= calc.computeAverageLocalOfObservations()

            if self.use_correction:
                marginal_mi = [ calc.computeAverageLocalOfObservations()
                                for calc in self.xmiCalcs ]
                psi += (self.N - 1) * np.min(marginal_mi)

        return psi


    def update_and_compute(self, X: Iterable[np.ndarray], V: np.ndarray) -> float:
        """
        """
        psi = PSI_START
        if not self.is_initialised:
            self.initialise_calculators(X, V)

        else:
            self.update_calculators(V)
            if self.sample_counter > self.sample_threshold:
                psi = self.compute_psi(V)

        self.past_X = X
        self.past_V = V
        self.sample_counter += 1

        self.past_psi_vals.append(psi)
        if len(self.past_psi_vals) > self.psi_buffer_size:
            self.past_psi_vals.pop(0)

        if self.use_filter:
            psi_filt = np.nanmedian(self.past_psi_vals)
            return psi_filt
        else:
            return psi


    def exit(self) -> None:
        """
        Gracefully shut down JVM. Call whenever done with the calculator.
        """
        if jp.isJVMStarted():
            print('Shutting down JVM...')
            jp.shutdownJVM()



def plot_psi(
        ts: np.ndarray, psis: np.ndarray, use_correction: bool,
        use_local: bool, use_filter: bool, obs_win: int, show: bool
    ) -> None:

    label = 'Using local MI' if use_local else 'Using average MI'
    label += ' and all past states' if obs_win <= 0 else f' and {obs_win} past states'
    if use_filter:
        label += ' (median filtered)'
    ylabel = '(1,1)' if use_correction else '(1)'
    ylabel = '$\\Psi^{' + ylabel + '}(\\tilde{x})$'

    fig = plt.figure()
    plt.plot(ts, psis, label = label)
    plt.xlabel('t (s)')
    plt.ylabel(ylabel)
    plt.legend()

    if show:
        plt.show()

    return fig


@click.command()
@click.option('--model',  help = 'Directory where system trajectories are stored for the model')
@click.option('--use-correction', is_flag = True, default = True,
              help = 'If true, use first-order lattice correction for emergence calculation.')
@click.option('--use-local',  is_flag = True, default = False,
              help = 'If true, use first-order lattice correction for emergence calculation.')
@click.option('--use-filter', is_flag = True, default = True,
              help = 'If true, apply median filter to the last n computations')
@click.option('--filter-buffer', default = 5,
              help = 'Number of timesteps to filter over')
@click.option('--observation-window', default = -1,
              help = 'Number of timesteps used for calculating Psi. Use all past data if <= 0.')
@click.option('--threshold',
              help = 'Number of timesteps to wait before calculation, at least as many as the dimenstions of the system')
def test(model: str, use_correction: bool, use_local: bool, use_filter: bool,
        filter_buffer: int, observation_window: int, threshold: int) -> None:
    """
    Test the emergence calculator on the trajectories specified in `filename`.
    """

    ts   = []
    psis = []
    thres = 0

    if model:
        m = FlockModel.load(model)
        X = m.traj['X']
        V = np.array([ centre_of_mass(Xi, m.l, m.bounds) for Xi in X ])
        ts = np.arange(len(X)) / m.dt

        calc = EmergenceCalculator(
            use_correction, use_filter, filter_buffer, observation_window, use_local)
        for i in range(len(X)):
            psi = calc.update_and_compute(X[i], V[i])
            if psi:
                psis.append(psi)
        thres = len(X[0]) + filter_buffer

        calc.exit()
    else:
        calc = EmergenceCalculator(
            use_correction, use_filter, filter_buffer, observation_window, use_local)

        X = np.random.randn(100,10,2)
        V = np.mean(X, axis=1)
        ts = range(100)
        for i in ts:
            psi = calc.update_and_compute(X[i], V[i])
            psis.append(psi)

        thres = 10 + filter_buffer
        calc.exit()

    plot_psi(ts[thres:], psis[thres:], use_correction, use_local, use_filter, observation_window,
            show=True)


if __name__ == "__main__":
    test()
