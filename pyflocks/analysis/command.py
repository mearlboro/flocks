import os
import sys
from typing import List

import click
import numpy as np
from matplotlib import pyplot as plt
from pyflocks.analysis import plot, order
from pyflocks.analysis.emergence import JVM, MutualInfo, system
from pyflocks.analysis.ensemble import __find_sims, ensemble_avg
from pyflocks.analysis.order import EnumParams, param
from pyflocks.models.factory import FlockFactory
from pyflocks.util.util import load_var, save_param


@click.group(name='analysis')
def cli():
    pass


@cli.command()
@click.option('--path', default='out/txt/', help='Path to load model data from')
@click.option('--out', default='out/order/', help='Path to save order param data to')
@click.option('--name', default='Vicsek', help='Model type or experiment to load')
@click.option('--ordp', default='ALL', type=click.Choice(order.EnumParams.names()),
              help='Order parameter to study, all by default')
@click.option('--conp', '-p', default=['rho', 'eta'], multiple=True,
              help='Control parameters by which to aggregate simulations')
@click.option('--skip', default=500,
              help='Number of transient states to skip to compute only on steady states')
@click.option('--redo', is_flag=True,
              help='If data exists, recompute it, otherwise just redo plot')
def ensemble_command(
        path: str, out: str, name: str, ordp: str, conp: List[str], skip: int, redo: bool,
) -> None:
    """
    After a large number of simulations or experiment are run, we compute average
    order parameters on the trajectories, and then average again for all sims
    with the same parameters. Then plot the control parameter vs the averaged
    order parameters.

    Warn if an output folder from a single simulation or experiment.

    Resulting plots will be stored in out/plt

    Run from the root pyflocks/ folder

        python -m analysis.ensemble [flags]
    """

    if ordp == "ALL":
        ordps = order.EnumParams.members()[1:]
    else:
        ordps = [order.EnumParams[ordp]]
    conp = list(conp)

    print(f"Will calculate order parameters {ordps} for the given trajectories and control parameters {conp}")

    conp_str = '_'.join(conp)

    sims = __find_sims(path, name)
    if not len(sims.keys()):
        print(f'No directories of type {name} found in {path}')
        sys.exit(0)

    pltitle = sims[list(sims.keys())[0]].title
    plt.rcParams['figure.figsize'] = 10, 7

    for ordp in ordps:
        fname = f"{out}/{name}_{conp_str}_{str(ordp)}"
        if os.path.exists(f"{fname}.npy") and not redo:
            stats = np.load(f"{fname}.npy", allow_pickle=True).item()
        else:
            stats = ensemble_avg(sims, conp, ordp, skip)
            np.save(fname, stats)

        print(f"Saving figure to {out}")
        if len(conp) == 2:
            plot.aggregate_2param(name, stats, conp, ordp, pltitle)
        if len(conp) == 3:
            plot.aggregate_3param(name, stats, conp, ordp, pltitle)


@cli.command()
@click.option('--model', help = 'Directory where system trajectories are stored')
@click.option('--est', type = click.Choice([ 'Gaussian', 'Kraskov1', 'Kraskov2', 'Kernel']),
              help = 'Mutual Info estimator to use', required = True)
@click.option('--decomposition', is_flag = True, default = False,
              help = 'If true, decompose Psi into the synergy, redundancy, and correction.')
@click.option('--pointwise',  is_flag = True, default = False,
              help = 'If true, use pointwise mutual information for emergence calculation.')
@click.option('--skip', default = 0,
              help = 'Number of timesteps to wait before calculation e.g. for removing transients')
def emergence(model: str, est: str,
              decomposition: bool, pointwise: bool, skip: int
              ) -> None:
    """
    Test the emergence calculator on the trajectories specified in `filename`, or
    on a random data stream.
    """
    JVM.start()
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

    skip = int(skip)
    est = MutualInfo.get(est)
    dts = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    results = system(X[skip:], M[skip:], dts, est, pointwise, pth)
    for dt, e in zip(dts, results):
        print(f"{dt}: {results[0]} {results[1]}")
    JVM.stop()


@cli.command(name='order')
@click.option('--path', required = True, help = 'Path to load model data from')
@click.option('--out',  required = True, help = 'Path to save data to', default = 'out/order/')
@click.option('--ordp', default = '', help = 'Order parameter to compute, all by default',
              type = click.Choice(EnumParams.names()))
@click.option('--redo', default = False,
              help = 'If data exists, recompute it, otherwise just redo plot')
def order_command(path: str, out: str, ordp: str, redo: bool) -> None:
    """
    After a simulation or experiment is run, compute (and plot) the results by
    showing trajectories, order parameters, susceptibilities, and histograms for the
    most 'interesting' states of that single run.

    It is assume that the simulation has a directory consistent with the mkdir
    method of the Flock abstract class, i.e. the dirname begins with the model
    name, followed by underscore, and other model details

        {name}(_{info})+(_{paramname}{paramval})+_{seed}?-id

    Run from the root pyflocks/ folder, will save the order parameters as CSV
    files in out/order and the plots in out/plt.

        python -m analysis.order [flags]
    """
    exp = FlockFactory.load(path)
    parampth = exp.mkdir(out)

    ords = dict()
    if ordp:
        ordp = EnumParams[ordp]
    else:
        ordp = EnumParams.ALL

    print(f"Computing order parameter(s) {ordp} for {path}, saving to {parampth}")

    if ordp != EnumParams.ALL:
        if os.path.exists(f"{parampth}/{ordp}.txt") and not redo:
            ords = { ordp: load_var(f"{parampth}/{ordp}.txt") }
        else:
            ords = param(ordp, exp.traj['X'], exp.traj['A'], exp.l, 0, exp.bounds)
            save_param(ords[ordp], str(ordp), parampth)
    else:
        for ordp in EnumParams.members()[1:]:
            if os.path.exists(f"{parampth}/{ordp}.txt") and not redo:
                ords[ordp] = load_var(f"{parampth}/{ordp}.txt")
            else:
                ords |= param(ordp, exp.traj['X'], exp.traj['A'], exp.l, 0, exp.bounds)
                save_param(ords[ordp], str(ordp), parampth)

    plot.order_params(exp, ords)
    # TODO: peak detection, plot those states
    plot.states([ 0, 50, 100, 150, 200, 300, 400, 499 ], exp)
