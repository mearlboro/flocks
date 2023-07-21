import click
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import sys

from analysis import order, plot
from flock.model import Flock
from flock.factory import FlockFactory

from typing import Any, Dict, List, Tuple


def __find_sims(path: str, name: str) -> Dict[str, 'Flock']:
    """
    Go through directory at `path` and filter by the model name `name` to load
    simulation results from disk into Flock objects for analysis.
    If `path` contains multiple directories with simulation results, then load
    all and return as a dict. Otherwise, check if the `path` is itself a
    simulation dir and if it is, load it.
    """
    d = os.path.basename(path)
    if name in d:
        return { d: FlockFactory.load(path) }

    dirs = [d for d in os.listdir(path)
              if os.path.isdir(os.path.join(path, d)) and d.lower().startswith(name.lower()) ]
    models = { d: FlockFactory.load(os.path.join(path, d)) for d in dirs }
    return models



def ensemble_avg(
        models:         Dict[str, Any],
        control_params: List[str],
        order_param:    str,
        tt:             int,
        path:           str = 'out/order'
    ) -> Dict[float, Dict[float, Any]]:
    """
    For all models loaded in the given dict, compute order params, and aggregate
    them by the control parameters by averaging all values for all steady states
    of one system and further grouping and averaging over systems with the same
    parameters into ensembles.

    We assume that for a given system it takes a number tt of timesteps to reach
    a steady state. This analysis will not be relevant if tt is small.

    Params
    ------
    models
        as returned by __find_sims
    control_params
        the list of parameters by which to aggregate stats, must be included in
        the param hash of the Flock object
    order_param
        order parameter names to compute and aggregate for the models
    tt
        number of timesteps to discard at the beginning of the simulation before
        computing the order parameters

    Returns
    ------
    statistics data from all experiments stored in models dict, in the form of
    nested dicts aggregated by values of control params as keys. Values are
    np.arrays of shape (T,) where each time point produces a statistic, otherwise
    scalars

    Example: Vicsek with control params rho and eta, will return
        Dict[float, Dict[float, Any]]

        stats = { 1.0: { 0.1: { 'Vicsek_order': [ ... ],
                                'var_angle'   : [ ... ], ...

    these dicts can be plugged directly into the 2param and 3param plotting
    functions
    """
    # extract the model hyper parameter names (e.g. eta, rho in Vicsek) and
    # check if the params passed to the function are correct
    model_params = set([p for m in models.values()
                          for p in m.params.keys() ])
    for p in control_params:
        if p not in model_params:
            raise ValueError(
                f"Parameter {p} passed to `aggregate` not in model params")
            exit(0)

    # group experiments with the same params into batches
    batch_names = set([ '_'.join(m.split('_')[:-1]) for m in models.keys() ])
    print(batch_names)

    stats = dict()
    for batch in batch_names:
        exp_in_batch = sorted([ m for m in models.keys() if batch in m ])
        count = len(exp_in_batch)
        print(f"Processing {count} experiments for {batch} with params " +
            " ".join([ f"{p}: {models[exp_in_batch[0]].params[p]}" for p in control_params ]) )

        # build a hash of hashes grouping model parameters by the param_list
        tmp = stats
        for p in control_params:
            pval = models[exp_in_batch[0]].params[p]
            if pval not in tmp.keys():
                tmp[pval] = dict()
            tmp = tmp[pval]

        # iterate through all time series from all experiments with those params
        # and collect statistics - averaged accross time foreach experiment
        for exp in exp_in_batch:
            m = models[exp]
            if 'X' not in m.traj.keys() or 'A' not in m.traj.keys():
                print(f"Error with experiment {exp}: incomplete data")
                continue

            Xt = m.traj['X'][tt:]
            At = m.traj['A'][tt:]

            Vt = order.param(order_param, Xt, At, m.l, m.params['r'], m.bounds)[order_param]
            np.save(f'{path}/{m.string}_{str(order_param)}', Vt, allow_pickle=True)

            if order_param in tmp.keys():
                tmp[order_param].append(np.mean(Vt))
            else:
                tmp[order_param] = [ np.mean(Vt) ]

        # average everything accross all experiments
        tmp[f'{order_param}_mean'] = np.mean(tmp[order_param])
        tmp[f'{order_param}_std']  = np.std( tmp[order_param])

    return stats


@click.command()
@click.option('--path', default='out/txt/',   help='Path to load model data from')
@click.option('--out', default='out/order/', help='Path to save model data to')
@click.option('--name', default='Vicsek',     help='Model type or experiment to load')
@click.option('--ordp', default='ALL', type=click.Choice(order.EnumParams.names()),
              help='Order parameter to study, all by default')
@click.option('--conp', '-p', default=[ 'rho', 'eta' ], multiple = True,
              help='Control parameters by which to aggregate simulations')
@click.option('--skip', default = 500,
              help='Number of transient states to skip to compute only on steady states')
@click.option('--redo', is_flag=True, default=False,
              help = 'If data exists, recompute it, otherwise just redo plot')
def main(
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

    sims = __find_sims(path, name)
    if not len(sims.keys()):
        print(f'No directories of type {name} found in {path}')
        sys.exit(0)

    if ordp == 'ALL':
        ordps = order.EnumParams.members()
    else:
        ordps = [ order.EnumParams[ordp] ]
    conp_str = '_'.join(conp)

    pltitle = sims[list(sims.keys())[0]].title
    plt.rcParams['figure.figsize'] = 10, 7

    for ordp in ordps:
        fname = f"{out}/{name}_{conp_str}_{str(ordp)}"
        if os.path.exists(f"{fname}.npy") and not redo:
            stats = np.load(f"{fname}.npy", allow_pickle = True).item()
        else:
            stats = ensemble_avg(sims, conp, ordp, skip)
            np.save(fname, stats)

        print(f"Saving figure to {out}")
        if len(conp) == 2:
            plot.aggregate_2param(name, stats, conp, ordp, pltitle, out)
        if len(conp) == 3:
            plot.aggregate_3param(name, stats, conp, ordp, pltitle, out)


if __name__ == "__main__":
    main()
