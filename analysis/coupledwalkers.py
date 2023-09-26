import click
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Callable, Dict, Iterable, List, Tuple, Union, NamedTuple

from analysis import order
from analysis.emergence import JVM, EmergenceCalc, MutualInfo, system, ensemble
from flock.model import Flock, FlockModel
from flock.factory import FlockFactory
from util import util


def rw_ensemble(
        sims: int, N: int, e: float, g: float, a: float, t: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from flock.walker import RandomWalker

    if g:
        X0  = np.array([i for i in range(N)]).reshape((N, 1)) * a
        rws = [ RandomWalker(seed = i, n = N, e = e, g = g, dx = 0, start_state = X0)
                for i in range(sims) ]
    else:
        rws = [ RandomWalker(seed = i, n = N, e = e, g = g, dx = 0, rand_state = True)
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
    pth = f"rwint/{name}"

    if not os.path.isdir(pth):
        os.mkdir(pth)

    # stationary case with additive time difference
    dts = [ 1, 2, 4, 8 ]
    t = 1000
    a = 1
    Xs, Vs = rw_ensemble(r, n, e, g, a, t)
    ensemble(True, Xs, Vs, dts, MutualInfo.ContinuousGaussian, path = pth)


if __name__ == "__main__":
    JVM.start()
    main()
    JVM.stop()
