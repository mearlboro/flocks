import os
from typing import Any, Dict, List

import numpy as np
from pyflocks.analysis import order
from pyflocks.models.factory import FlockFactory
from pyflocks.models.model import Flock
from pyflocks.util.util import load_var, save_param


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
        return {d: FlockFactory.load(path)}

    dirs = [d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.lower().startswith(name.lower())]
    models = {d: FlockFactory.load(os.path.join(path, d)) for d in dirs}
    return models


def ensemble_avg(
        models: Dict[str, Any],
        control_params: List[str],
        order_param: str,
        skip: int,
        path: str = 'out/order'
) -> Dict[float, Dict[float, Any]]:
    """
    For all models loaded in the given dict, compute order params, and aggregate
    them by the control parameters by averaging all values for all steady states
    of one system and further grouping and averaging over systems with the same
    parameters into ensembles.

    We assume that for a given system it takes a number skip of timesteps to reach
    a steady state, so before averaging the instantaneous order parameter at each
    time step we remove the first skip.

    Params
    ------
    models
        as returned by __find_sims
    control_params
        the list of parameters by which to aggregate stats, must be included in
        the param hash of the Flock object
    order_param
        order parameter names to compute and aggregate for the models
    skip
        number of timesteps to discard at the beginning of the simulation before
        computing the mean of order parameters
    path
        where to save order parameter computations.

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
                        for p in m.params.keys()])
    for p in control_params:
        if p not in model_params:
            raise ValueError(
                f"Parameter {p} passed to `aggregate` not in model params")
            exit(0)

    # group experiments with the same params into batches
    batch_names = set(['_'.join(m.split('_')[:-1]) for m in models.keys()])
    print(batch_names)

    stats = dict()
    for batch in batch_names:
        exp_in_batch = sorted([m for m in models.keys() if batch in m])
        count = len(exp_in_batch)
        print(f"Processing {count} experiments for {batch} with params " +
              " ".join([f"{p}: {models[exp_in_batch[0]].params[p]}" for p in control_params]))

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

            Xt = m.traj['X']
            At = m.traj['A']

            parampth = m.mkdir(path)
            if os.path.exists(f"{parampth}/{str(order_param)}.txt"):
                Vt = load_var(f"{parampth}/{str(order_param)}.txt")
            else:
                Vt = order.param(order_param, Xt, At, m.l, m.params['r'], m.bounds)[order_param]
                save_param(Vt, str(order_param), parampth)

            if order_param in tmp.keys():
                tmp[order_param].append(np.mean(Vt[skip:]))
            else:
                tmp[order_param] = [np.mean(Vt[skip:])]

        # average everything accross all experiments
        tmp[f'{order_param}_mean'] = np.mean(tmp[order_param])
        tmp[f'{order_param}_std'] = np.std(tmp[order_param])

    return stats
