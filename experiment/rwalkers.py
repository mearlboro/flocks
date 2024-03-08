"""
This script is used to compute the Psi-Gamma-Delta theory of emergence on
INTERACTING Gaussian random walkers using FIXED time index t=1, varying t'
and assuming NON-STATIONARITY
"""

import click
import numpy as np
import os

from typing import Callable, Dict, Iterable, List, Tuple, Union, NamedTuple

from analysis.emergence import JVM, EmergenceCalc, MutualInfo


def rw_ensemble(
        sims: int, N: int, e: float, g: float, a: float, t: int,
        seeds: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from flock.walker import RandomWalker

    if g:
        X0  = np.array([i for i in range(N)]).reshape((N, 1)) * a
        rws = [ RandomWalker(seed = seeds[i], n = N, e = e, g = g, dx = 0, start_state = X0)
                for i in range(sims) ]
    else:
        rws = [ RandomWalker(seed = seeds[i], n = N, e = e, g = g, dx = 0, rand_state = True)
                for i in range(sims) ]

    for _ in list(range(t)):
        for rw in rws:
            rw.update()

    Xs = np.array([ np.array(rw.traj['X']) for rw in rws ])
    Vs = np.array([ np.mean(X, axis = 1)   for X  in Xs  ])
    return Xs, Vs


@click.command()
@click.option('-s', default = 0)
@click.option('-n', default = 32)
@click.option('-e', default = 1.0)
@click.option('-g', default = 0.0)
@click.option('-r', default = 2000)
def main(s: int, n: int, e: float, g: float, r: int):
    title = f"{n} random walkers with coupling $\\gamma$ = {g} and noise " + \
             "$\\eta \sim \mathcal{N}$" + f"(0, {e})"
    title += f"\n(ensembles of {r} realisations)"
    name = f"RandomWalkers_ens{r}_n{n}_e{e}_g{g}_dx0__t1-"
    pth = f"out/rw/{name}"

    if not os.path.isdir(pth):
        os.mkdir(pth)

    if s:
        np.random.seed(s)
        seeds = np.random.randint(r * 10, size = r)
    else:
        seeds = range(r)

    # stationary case with additive time difference
    dts = range(1,21)
    t = 25
    a = 1

    Xs, Vs = rw_ensemble(r, n, e, g, a, t, seeds)

    for dt in dts:
       print(f"Computing emergence from t={1} to t'={1+dt}")
       X  = np.concatenate((
             Xs[:, 0, :].reshape(r, n, 1),
             Xs[:, 0 + dt, :].reshape(r, n, 1)))
       V  = np.concatenate((
             Vs[:, 0].reshape(r, 1),
             Vs[:, 0 + dt].reshape(r, 1)))

       EmergenceCalc(X, V, MutualInfo.ContinuousGaussian, False,
                     r, f"{pth}{1+dt}")

if __name__ == "__main__":
    JVM.start()
    main()
    JVM.stop()
