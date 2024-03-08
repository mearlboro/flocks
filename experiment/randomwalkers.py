"""
This script was used to compute the original analysis of Psi-Gamma-Delta
theory of emergence on NON-INTERACTING random walkers using MULTIPLICATIVE
time index tau=t'/t assuming NON-STATIONARITY
"""


import click
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Callable, Dict, Iterable, List, Tuple, Union, NamedTuple

from analysis import order
from analysis.emergence import JVM, EmergenceCalc, MutualInfo
from util import util

def plot_emergence(
        dts: np.ndarray, emstats: np.ndarray, mistats: np.ndarray, n: int,
        decomposition: bool = False,
        axs = None, path: str = '', title: str = '', xlab: str = '',
        lab: str = '', col: str = '', fmt: str = ''
    ):
    plt.cla()
    plt.rcParams['figure.figsize'] = 12, 24

    # get theoretical quantities
    thvv = [ 1/2 * np.log(dt/(dt - 1)) for dt in dts ]
    thxv = [ n/2 * np.log(n*dt/(n*dt - 1)) for dt in dts ]
    thp  = [ vv - xv for (vv, xv) in zip(thvv, thxv) ]
    #thd  = [ -1/2 * np.log(1 + (n - 1)/(dt - 1)) + (n - 1)/dt for dt in dts ]
    #thg  = [ -1/2 * np.log(1 + (n - 2)/dt) + (n - 1)/dt for dt in dts ]

    if axs is None:
        nplots = 4
        if decomposition:
            nplots += 5
        _, axs = plt.subplots(nplots, 1)
        plt.suptitle(title)

    # psi k=0
    axs[0].errorbar(dts, emstats[:, 0, 0], yerr = emstats[:, 1, 0],
        linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
    axs[0].set_ylabel("$\\Psi^{(k,0)}_{t,t'} (V)$\n(no correction)")
    axs[0].plot(dts, thp, color = 'black', linestyle=':', linewidth=1)
    # psi k=1
    axs[1].errorbar(dts, emstats[:, 0, 1], yerr = emstats[:, 1, 1],
        linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
    axs[1].set_ylabel("$\\Psi^{(k,1)}_{t,t'} (V)$\n(k=1 correction)")

    if decomposition:
        # whole
        axs[2].plot(dts, thvv, color = 'black', linestyle=':', linewidth=0.5)
        axs[2].errorbar(dts, emstats[:, 0, 2], yerr = emstats[:, 1, 2],
            linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
        axs[2].set_ylabel(f"$I(V(t); V(t'))$")
        # parts
        axs[3].plot(dts, thxv, color = 'black', linestyle=':', linewidth=0.5)
        axs[3].errorbar(dts, emstats[:, 0, 3], yerr = emstats[:, 1, 3],
            linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
        axs[3].set_ylabel(f"$\\sum_i I(X_i(t); V(t'))$")
        # correction
        axs[4].errorbar(dts, emstats[:, 0, 4], yerr = emstats[:, 1, 4],
            linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
        axs[4].set_ylabel(f"(n-1)$min_i (I(X_i(t);V(t')))$")

    #axs[-2].plot(dts, thd, color = 'black', linestyle=':', linewidth=0.5)
    axs[-2].errorbar(dts, dstats[:, 0], yerr = dstats[:, 1],
            linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
    axs[-2].set_ylabel("$\\Delta^{(1)}_{t,t'} (V)$")
    #axs[-1].plot(dts, thg, color = 'black', linestyle=':', linewidth=0.5)
    axs[-1].errorbar(dts, gstats[:, 0], yerr = gstats[:, 1],
            linewidth = 1, fmt = fmt, color = col, label = lab, linestyle = '--')
    axs[-1].set_ylabel("$\\Gamma^{(1)}_{t,t'} (V)$")
    for ax in axs:
        ax.set_xlabel(xlab)
        ax.set_xticks(dts[::2])

    plt.tight_layout()
    if path:
        plt.savefig(f"{path}/emergence.pdf", dpi = 320)

class MutualInfos(NamedTuple):
    vmi: float
    xvmi: List[float]
    vxmi: List[float]
    xiximi: List[float]
    xixjmi: List[float]

class MutualInfoStats(NamedTuple):
    vmi: float
    xvmi: float
    vxmi: float
    xiximi: float
    xixjmi: float

class EmergenceStats(NamedTuple):
    psik0: float
    psik1: float
    gamma: float
    delta: float

milabels = [
    "$I(V(t); V(t'))$",
    "$I(X_i(t); V(t'))$",
    "$I(V(t); X_i(t'))$",
    "$I(X_i(t); X_i(t'))$",
    "$I(X_i(t); X_j(t'))$",
]
emlabels = [
    "$\\Psi^{(k,0)}_{t,t'} (V)$\n(no correction)",
    "$\\Psi^{(k,1)}_{t,t'} (V)$\n(k=1 correction)",
    "$\\Gamma^{(1)}_{t,t'} (V)$",
    "$\\Delta^{(1)}_{t,t'} (V)$",
]

def calc_for_system(X, V, dt):
    calc = EmergenceCalc(X, V, MutualInfo.ContinuousGaussian,
                            pointwise = False, dt = dt)

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
                                         for i in range(calc.n) ]
    )
    return e, mi


def nonstationary_ensemble(
        Xs: np.ndarray, Vs: np.ndarray, dts: np.ndarray, n: int,
        path: str = ''
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    For non-stationary time series, we cannot directly apply MI between Xi and
    V at arbitrary times, due to the distribution of each Xi and V being dependent
    on t. Instead, we use an ensemble, and take all Xi at the same time t, and V
    at the time t'. We group results by t'/t rather than t'-t.

    Xs of shape (S, T, N) are ensembles for S simulations, T timesteps, N variables
    Vs of shape (S, T, 1) are values of the macroscopic variable over all N

    an ensemble X[sim_number, t, i] I can apply MI on X[:, t, :], X[:, t', :] to get the value for t'-t.
aaaand id be down to h    will it be the
    '''

    emstats, mistats = [], []
    mean_std = lambda xs: (np.mean(xs, axis = 0), np.std(xs, axis = 0))

    sims, T, N = Xs.shape[:3]
    repeat = int(T / (max(dts) - 1))
    print(f"Computing emergence for {repeat} pairs of t,t'")

    for dt in dts:
        ems, mis = [], []
        # We get all points Xi at fixed t = t1 for and V at t' = dt * t1
        # to compute non-linear time differences, we construct X and V arrays
        # which contain the values at both times t and t'
        for t in range(1, repeat):
            print(f"Computing emergence from t={t} to t'={t*dt}")
            X  = np.concatenate((
                    Xs[:,    t, :].reshape(sims, N),
                    Xs[:, t*dt, :].reshape(sims, N)))
            V  = np.concatenate((
                    Vs[:,    t].reshape(sims),
                    Vs[:, t*dt].reshape(sims)))

            e, mi = calc_for_system(X, V, sims)
            ems.append(e)
            mis.append(mi)

        # since every dt and t produces a single value, the mean and std are over t
        emstats.append(mean_std(np.array(ems))) # has shape (2, 4)
        # for the individual MI over the mean and std should be computed over i
        # first get means over t, and the mean and std dev of those are over i
        mis_stats = np.array([
            ( np.mean([mis[t].vmi    for t in range(0, repeat - 2)]), 0 ),
            mean_std(np.mean([mis[t].xvmi   for t in range(0, repeat - 2)], axis = 0)),
            mean_std(np.mean([mis[t].vxmi   for t in range(0, repeat - 2)], axis = 0)),
            mean_std(np.mean([mis[t].xiximi for t in range(0, repeat - 2)], axis = 0)),
            mean_std(np.mean([mis[t].xixjmi for t in range(0, repeat - 2)], axis = 0))
            ]).T
        mistats.append(mis_stats) # has shape (2, 5)

    if path:
        np.save(f"{path}/ens_emstats", np.array(emstats)) # has shape (dts, 2, 4)
        np.save(f"{path}/ens_mistats", np.array(mistats)) # has shape (dts, 2, 5)

    return emstats, mistats


cols = [ 'blue', 'turquoise', 'green', 'orange', 'salmon', 'red', 'purple', 'violet' ]
fmts = [ 's', 'o', '.', 'x', '+', '^', 'v' ]
def plot_ensembles_multi_n(dts, sims = 10000,
        ns: List[int] = [ 4, 8, 16, 32, 64 ]):

    axs = None

    for i,n in enumerate(ns):
        path = f"rw/RandomWalkers_ens{sims}_n{n}_e1.0_dt0"
        emstats = np.load(f"{path}/ens_emstats.npy", allow_pickle = True)
        mistats = np.load(f"{path}/ens_mistats.npy", allow_pickle = True)

        axs = plot_emergence(dts, emstats, mistats, n,
            decomposition = True, axs = axs, path = '', title = '', xlab = "$t'/t$")

    plt.legend(loc='upper center', ncol = 5)
    plt.savefig(f"combined_data.pdf", dpi = 360)




def rw_ensemble(
        sims: int, N: int, e: float, g: float, t: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from flock.walker import RandomWalker

    X0  = np.array([i for i in range(N)]).reshape((N, 1))
    rws = [ RandomWalker(seed = i, n = N, e = e, g = g, dx = 0, start_state = X0)
            for i in range(sims) ]
    for _ in list(range(t)):
        for rw in rws:
            rw.update()

    Xs = np.array([ np.array(rw.traj['X']) for rw in rws ])
    Vs = np.array([ np.mean(X, axis = 1)   for X  in Xs  ])
    return Xs, Vs


@click.command()
@click.option('-n', default = 32)
@click.option('-e', default = 1.0)
@click.option('-g', default = 0.0)
@click.option('-r', default = 1000)
def main(n: int, e: float, g: float, r: int):
    title = f"{n} random walkers with coupling $\\gamma$ = {g} and noise " + \
             "$\\eta \sim \mathcal{N}$" + f"(0, {e})"
    title += f"\n(ensembles of {r} realisations)"
    name = f"RandomWalkers_ens{r}_n{n}_e{e}_g{g}_dx0"
    pth = f"rw/{name}"

    dts = range(2, 21)
    t = 200

    Xs, Vs = rw_ensemble(r, n, e, g, t)
    if not os.path.isdir(pth):
        os.mkdir(pth)
    nonstationary_ensemble(Xs, Vs, dts, n, pth)


if __name__ == "__main__":
    JVM.start()
    main()
    JVM.stop()
