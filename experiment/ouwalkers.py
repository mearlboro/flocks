"""
This script is used to compute the Psi-Gamma-Delta theory of emergence on
INTERACTING Ornstein-Uhlenbeck random walkers using ADDITIVE time index
dt=t'-t assuming STATIONARITY
"""


import click
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Callable, Dict, Iterable, List, Tuple, Union, NamedTuple

from analysis.emergence import JVM, EmergenceCalc, MutualInfo, system, ensemble

def rw_ensemble(
        sims: int, N: int, e: float, k: float, g: float, t: int,
        seeds: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from flock.ornstein import OUWalker

    rws = [ OUWalker(seed = seeds[i], n = N, e = e, k = k, g = g, dx = 0,
                        rand_state = False, start_state = None)
            for i in range(sims) ]

    for _ in list(range(t)):
        for rw in rws:
            rw.update()

    Xs = np.array([ np.array(rw.traj['X']) for rw in rws ])
    Vs = np.array([ np.mean(X, axis = 1)   for X  in Xs  ])
    return Xs, Vs


@click.command()
@click.option('-s', default = 0  ,  help="Seed")
@click.option('-n', default = 32 ,  help="Number of walkers")
@click.option('-e', default = 0.5,  help="Std dev of Gaussian noise")
@click.option('-k', default = 0.01, help="Restorative force")
@click.option('-g', default = 0.4,  help="Coupling coefficient")
@click.option('-r', default = 10 ,  help="Realisations")
def main(s: int, n: int, e: float, k: float, g: float, r: int):
    name = f"OUWalkers_stat_ens{r}_n{n}_e{e}_k{k}_g{g}_dx0_dt"
    pth = f"out/ou/{name}"

    if not os.path.isdir('ou'):
        os.mkdir('ou')
    if not os.path.isdir(pth):
        os.mkdir(pth)

    if s:
        np.random.seed(s)
        seeds = np.random.randint(r * 10, size = r)
    else:
        seeds = range(r)

    # stationary case with additive time difference
    dts = [ 1, 2, 4, 8 ]
    t = 10000
    Xs, Vs = rw_ensemble(r, n, e, k, g, t, seeds)

    for dt in dts:
        for i in range(r):
            EmergenceCalc(Xs[i], Vs[i], MutualInfo.ContinuousGaussian,
            False, dt, f"{pth}{dt}-seed{seeds[i]}")

if __name__ == "__main__":
    JVM.start()
    main()
    JVM.stop()
