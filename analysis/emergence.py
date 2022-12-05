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

from typing import Callable, Dict, Iterable, List, Tuple, Union

from flock.model import FlockModel
from util.geometry import centre_of_mass
from util import util


INFODYNAMICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'infodynamics.jar')
PSI_START = -5

def startJVM() -> None:
    """
    Start Java Virtual Machine to run Java code inside this Python repository
    """
    if not jp.isJVMStarted():
        print('Starting JVM...')
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s'%INFODYNAMICS_PATH)

def stopJVM() -> None:
    """
    Gracefully exit Java Virtual Machine
    """
    if jp.isJVMStarted():
        print('Shutting down JVM...')
        jp.shutdownJVM()


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
            decomposition: bool = True,
            use_correction: bool = True,
            use_filter: bool = True,
            psi_buffer_size : int = 20,
            observation_window_size : int = -1,
            use_local : bool = True,
            sample_threshold: int = 0,
        ) -> None:
        """
        Construct the emergence calculator by setting member variables and
        checking the JVM is started. The JIDT calculators will be initialised
        later, when the first batch of data is provided.

        After calculating the value of emergence for a given frame, it is
        median-filtered with recent past values to reduce volatility.

        Parameters
        ----------
        decomposition : bool
            Whether to return both the synergy and redundancy, or just the
            synergy-redundancy index (i.e. Psi = syn - red)
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

        self.decomposition = decomposition
        self.use_filter = use_filter
        self.use_correction = use_correction
        self.psi_buffer_size = psi_buffer_size
        self.sample_threshold = sample_threshold

        self.past_psi_vals = [PSI_START] * psi_buffer_size

        self.observation_window_size = observation_window_size

        self.observations_V = []
        self.observations_X = []

        self.use_local = use_local

        startJVM()

        print(f'Successfully initialised EmergenceCalculator.')
        print(f' decomposition:  {decomposition}')
        print(f' observations:   {observation_window_size}')
        print(f' use correction: {use_correction}')
        print(f' use local:      {use_local}')
        print(f' use filter:     {use_filter}')
        if use_filter:
            print(f' buffer: {psi_buffer_size}')


    def initialise_calculators(self, X: np.ndarray, V: np.ndarray) -> None:
        """
        Initialise calculators of mutual information for the number of
        variables in the current system.

        Params
        ------
        X
            system micro variables of shape (N, T) for N components in the
            system
        V
            candidate emergence feature: D-dimensional system macro variable of
            shape (D, T)
        """
        self.N = len(X)
        if self.sample_threshold < self.N:
            self.sample_threshold = self.N
        print(f' threshold:      {self.sample_threshold}')

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
        Add system trajectories to the mutual information calculators so that
        the distributions of the system variables can be estimated for computing
        the mutual information.

        Params
        ------
        V
            candidate emergence feature: D-dimensional system macro variable of
            shape (D, T)
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


    def compute_psi(self, V: np.ndarray) -> Tuple[float, float, float]:
        """
        Using the observations stored in the mutual information calculators,
        estimate distributions and mutual information between micro variables
        Xi at different time instants, as well as the mutual information
        between them and the macroscopic variable V, and between V itself at
        different time instants.

        The number of past values is given by `observation_window`.

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
        self.vmiCalc.finaliseAddObservations()
        jV = javify(V)

        syn, red, corr = 0, 0, 0
        if self.use_local:
            syn = self.vmiCalc.computeLocalUsingPreviousObservations(
                    javify(self.past_V), jV)[0]
            red = 0
            for Xip,calc in zip(self.past_X, self.xmiCalcs):
                calc.finaliseAddObservations()
                red += calc.computeLocalUsingPreviousObservations(javify(Xip), jV)[0]
            if self.use_correction:
                marginal_mi = [ calc.computeAverageLocalOfObservations()
                                for calc in self.xmiCalcs ]
                corr += (self.N - 1) * np.min(marginal_mi)

        else:
            syn = self.vmiCalc.computeAverageLocalOfObservations()
            for calc in self.xmiCalcs:
                calc.finaliseAddObservations()
                red += calc.computeAverageLocalOfObservations()
            if self.use_correction:
                marginal_mi = [ calc.computeAverageLocalOfObservations()
                                for calc in self.xmiCalcs ]
                corr += (self.N - 1) * np.min(marginal_mi)

        return syn, red, corr


    def update_and_compute(self,
            X: Iterable[np.ndarray], V: np.ndarray
        ) -> Union[float, Tuple[float, float, float]]:
        """
        Add observations to calculators (using update_calculators) then
        compute synergy and redundancy, and return Psi as specified in
        the class parameters.

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
        if `decomposition` is set, 3 floats representing synergy, redundancy,
            and the correction.
        otherwise, returns one float, Psi = syn - red + corr, which may be
        median filtered if `filter` is set.
        if `use_correction` is not set, corr is always 0.
        """
        psi = PSI_START
        syn, red, corr = 0, 0, 0

        if not self.is_initialised:
            self.initialise_calculators(X, V)

        else:
            self.update_calculators(V)
            if self.sample_counter > self.sample_threshold:
                syn, red, corr = self.compute_psi(V)
                psi = syn - red + corr

        self.past_X = X
        self.past_V = V
        self.sample_counter += 1

        self.past_psi_vals.append(psi)
        if len(self.past_psi_vals) > self.psi_buffer_size:
            self.past_psi_vals.pop(0)

        if self.decomposition:
            return syn, red, corr
        if self.use_filter:
            psi_filt = np.nanmedian(self.past_psi_vals)
            return psi_filt
        else:
            return psi


def plot_psi(
        ts: np.ndarray, psis: np.ndarray, use_correction: bool,
        use_local: bool, use_filter: bool, obs_win: int, show: bool
    ) -> None:

    label = 'Using pointwise MI' if use_local else 'Using Shannon MI'
    label += ' and all past states' if obs_win <= 0 else f' and {obs_win} past states'
    if use_filter:
        label += ' (median filtered)'

    fig = plt.figure()

    if len(psis[0]) == 1:
        plt.plot(ts, psis, label = label)

        ylabel = '(1,1)' if use_correction else '(1)'
        ylabel = '$\\Psi^{' + ylabel + '}(\\tilde{x})$'
        plt.ylabel(ylabel)
    elif len(psis[0]) > 1:
        # the decomposition is given as opposed to the value so plot that
        syn = [ p[0] for p in psis ]
        red = [ p[1] for p in psis ]
        cor = [ p[2] for p in psis ]
        psi = [ s - r + c for (s, r, c) in psis ]

        plt.plot(ts, syn, label = 'Synergy', linewidth = 1)
        plt.plot(ts, red, label = 'Redundancy', linewidth = 1)
        if not all(c == 0 for c in cor):
            plt.plot(ts, cor, label = 'Lattice correction',
                     linewidth = 1, linestyle = 'dashed')
        plt.plot(ts, psi, label = 'Psi', linewidth = 2)

    plt.xlabel('t (s)')
    plt.legend()

    if show:
        plt.show()

    return fig


@click.command()
@click.option('--model',  help = 'Directory where system trajectories are stored for the model')
@click.option('--decomposition', is_flag = True, default = True,
              help = 'If true, decompose Psi into the synergy, redundancy, and correction.')
@click.option('--use-correction', is_flag = True, default = False,
              help = 'If true, use first-order lattice correction for emergence calculation.')
@click.option('--use-local',  is_flag = True, default = False,
              help = 'If true, use pointwise mutual information for emergence calculation.')
@click.option('--use-filter', is_flag = True, default = True,
              help = 'If true, apply median filter to the last n computations')
@click.option('--filter-buffer', default = 5,
              help = 'Number of timesteps to filter over')
@click.option('--observation-window', default = -1,
              help = 'Number of timesteps used for calculating Psi. Use all past data if <= 0.')
@click.option('--threshold',
              help = 'Number of timesteps to wait before calculation, at least as many as the dimenstions of the system')
def test(
        model: str, decomposition: bool, use_correction: bool, use_local: bool,
        use_filter: bool, filter_buffer: int, observation_window: int, threshold: int
    ) -> None:
    """
    Test the emergence calculator on the trajectories specified in `filename`, or
    on a random data stream.
    """

    ts   = []
    psis = []
    if not threshold:
        threshold = 0

    if model:
        m = FlockModel.load(model)
        X = m.traj['X']
        V = np.array([ centre_of_mass(Xi, m.l, m.bounds) for Xi in X ])
        ts = np.arange(len(X)) / m.dt

        calc = EmergenceCalculator(decomposition, use_correction, use_filter,
                filter_buffer, observation_window, use_local, threshold)
        for i in range(len(X)):
            psi = calc.update_and_compute(X[i], V[i])
            if psi:
                psis.append(psi)
        thres = len(X[0]) + filter_buffer
    else:
        calc = EmergenceCalculator(decomposition, use_correction, use_filter,
                filter_buffer, observation_window, use_local, threshold)

        X = np.random.randn(100,10,2)
        V = np.mean(X, axis=1)
        ts = range(100)
        for i in ts:
            psi = calc.update_and_compute(X[i], V[i])
            psis.append(psi)

        thres = 10 + filter_buffer

    plot_psi(ts[thres:], psis[thres:], use_correction, use_local, use_filter, observation_window,
            show=True)


if __name__ == "__main__":
    startJVM()
    test()
    stopJVM()
