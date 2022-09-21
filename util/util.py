#!/usr/bin/python
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from util.analysis import process_angles, process_space

from typing import Any, Dict, List


def sim_dir(root_dir: str, sim_dir: str) -> str:
    """
    Create a directory to store results of the current simulation. If it already
    exists, then add another one and return the path.

    Params
    -----
    root_dir { 'out/txt', 'out/img' }
        root directory for text dumps or plot exports
    sim_dir
        a directory for the current simulation, with a name of the form
            model_params-simID
    """

    pth = f'{root_dir}/{sim_dir}'

    while os.path.isdir(pth):
        # get the last sim ID
        if '-' in pth:
            [prefix, str_id] = pth.split('-')
            sim_id = int(str_id) + 1
            pth = f'{prefix}-{sim_id}'
        else:
            pth = f'{pth}-1'

    os.mkdir(pth)

    return pth


def dump_state(
        X: np.ndarray, filename: str, path: str
    ) -> None:
    """
    Dump the current state of the system variable passed as param to its
    corresponding file.

    Each system variable is dumped to an individual file correponding to that
    simulation.
    """
    n = len(X)

    with open(f'{path}/{filename}.txt', 'a') as f:
        for i in range(n):
            f.write( f'{X[i]}\t')

        f.write('\n')

    return


def save_var(X: np.ndarray, fn: str, path: str) -> None:
    """
    Append the variable `X` passed as param to the file `fn` in `path`.
    """
    n = len(X)

    with open(f'{path}/{fn}.txt', 'a') as f:
        for i in range(n):
            f.write( f'{X[i]}\t')

        f.write('\n')
    return


def load_var(filename: str) -> np.ndarray:
    """
    Load time series from file created with `dump_var` and return as numpy array.

    Each file contains a variable for an n-dimensional system, with the index i
    on columns and the time t on rows. Values are floats separated by tabs.

	A line represents a time step t. Position in D-dimensional space uses D
	columns, angular velocities and other scalars are one a single column

            x1  x2   ... xD
        t=1 .4  .7 . ... .3
        t=2 .1  ...  ...
        ...
        t=T ...

    The resulting array will have shape (T, D), with variable i at time t stored
    in X[t, i].
    """
    with open(filename, 'r') as f:
        X = [[x for x in line.split('\t') if x != '\n'] for line in f]

    return np.array(X).astype(float)


def position_matrix(Xs: List[np.ndarray]) -> np.ndarray:
    """
    Combine 2D numpy arrays representing individual coordinates in a
    D-dimensional space into 3D array

    Params
    ------
    Xs
        list of numpy arrays of shape (T, N)

    Returns
    ------
    numpy array of shape (T, N, D) for a D-dimensional system
    """
    (T, N) = Xs[0].shape

    return np.array([ np.stack([ X[i] for X in Xs ], axis = 1)
                      for i in range(T) ])


def load_model(path: str) -> Dict[str, Any]:
    """
    Load all N model variables from a given path after a simulation of T steps

    Params
    ------
    path
        path where files for the model variables are stored as TXT files, after
		being dumped step by step with `dump_state`, of the form

            out/txt/{model_specific_folder}

    Returns
    ------
    combines params dict with the variables loaded from file into a dictionary
    detailing the model

    variables are stored as numpy 2D array of shape (T, N) or (T, N, D), where
    X[t, i] or X[t, i, :] is the value of system variable Xi at time t

    for example, for a system with t timesteps, n variables, in 2 dimensions

        { 't' : 100, 'n' : 10, 'd' : 2,
          'X' : np.array(100,10,2), 'A': np.array(100,10) }
    """
    files = [f for f in os.listdir(path)
               if os.path.isfile(os.path.join(path, f))
			   and f.split('.')[-1] == 'txt' ]

    # loads all files into dict with filename (no extension, uppercase) as key
    var_dict = {}
    if files:
        var_dict = { f.split('.')[0].upper(): load_var(os.path.join(path, f))
                      for f in files }

    # variables named x or y denote coordinates and should be combined
    coord_vars = sorted([var for var in var_dict.keys()
                             if 'X' in var or 'Y' in var])
    pos_matrix = position_matrix([var_dict[k] for k in coord_vars])
    for k in coord_vars:
        del var_dict[k]
    var_dict['X'] = pos_matrix

    # extract number of particles and simulation timesteps from var dimension
    var_dict |= dict(zip(['t', 'n', 'd'], var_dict['X'].shape))

    return var_dict


def find_models(path: str, name: str) -> Dict[str, Dict[str, float]]:
    """
    Given a path, search for directories in it, and match their name to extract
    model names and parameters

    Params
    -----
    path
        Paths should be compatible with `sim_dir` paths, of the following form

            out/txt/{modelname}_({paramname}{paramvalue})+(_bounded)?

    name
        string to match model name

    Returns
    ------
    everything that matches as a dict, with the directory basename as key, and
    model name and parameters for that directory in another dict, for example

        { 'vicsek_eta1.0_rho1.11_bounded':
            { 'eta': 1.0, 'rho': 1.11, 'title': 'Vicsek  }
        }
    """

    dirs = [d for d in os.listdir(path)
              if os.path.isdir(os.path.join(path, d)) and name in d ]

    # split basename by _ to get name and params
    dirs_dict = { d: d.split('_') for d in dirs }
    # name and bounded are redundant, already included in the dir basename
    dirs_dict = { d: s[1:] for d,s in dirs_dict.items() }
    dirs_dict = { d: s[:-1] if 'bounded' in s[-1] else s for d,s in dirs_dict.items() }
    # numerical params have the form {string}{number}, split with regex
    model_dict = { d: { re.findall('[a-z]+', par)[0]: float(re.findall('[0-9.]+', par)[0])
                   for par in s if 'boundd' not in par }
                   for d,s in dirs_dict.items() }
    # construct plot-friendly model title
    for m in model_dict.keys():
        title = name
        if 'bounded' in m:
            title += ' bounded'
        for var in model_dict[m].keys():
            if len(var) > 1:
                title += f" $\\{var}$"
            else:
                title += f" ${var}$"
            title += f" = {model_dict[m][var]}"
        model_dict[m]['title'] = title

    return model_dict


def aggregate_model_stats(
        models: Dict[str, Any], path: str = 'out/txt/'
    ) -> Dict[float, Any]:
    """
    Load all models specified in the given dict, compute stats, and aggregate
    them by the parameters and save them into dicts

    Params
    ------
    models
        as returned by find_models

    Side-effects
    ------
    Updates the models param with the results of analysis and stats

    Returns
    ------
    statistics data from all experiments stored in models dict, in the form of
    nested dicts aggregated by model params as keys. Values are np.arrays of
    shape (T,)

    Example: Vicsek with params rho and eta, will return
        Dict[float, Dict[float, Any]]

        stats = { 1.0: { 0.1: { 'avg_dist_cmass': [ ... ],
                                'var_dist_cmass': [ ... ], ...

    """
    stats = dict()
    # we are interested in certain system stats aggregated over experiments
    stats_names = [ 'avg_dist_cmass', 'var_dist_cmass', 'avg_angle', 'var_angle',
                    'avg_abs_vel', 'var_abs_vel' ]

    # extract the model hyper parameter names (e.g. eta, rho in Vicsek)
    model_params = set([p for _,params in models.items()
                          for p in params if p != 'title'])

    # group experiments with the same params into batches
    # repeated experiments have '-' in name/path
    batch_names = sorted([ m for m,_ in models.items() if '-' not in m ])

    for batch in batch_names:
        exp_in_batch = sorted([ m for m in models.keys() if batch in m ])
        count = max([int(m.split('-')[1]) if len(m.split('-')) > 1 else 0
                        for m in exp_in_batch ]) + 1
        print(f"Processing {count} experiments for {batch} with params " +
            " ".join([ f"{p}: {models[batch][p]}" for p in model_params ]) )

        # Vicsek: we are interested in rho and eta as params
        # TODO: generalise so it works for other models
        rho = models[batch]['rho']
        eta = models[batch]['eta']
        if rho not in stats.keys():
            stats[rho] = dict()
        if eta not in stats[rho].keys():
            stats[rho][eta] = dict()

        # iterate throuh all time series from all experiments with those params
        # and collect statistics
        for exp in exp_in_batch:
            m = models[exp]
            m |= load_model(os.path.join(path, exp))

            (T, N, D)  =  models[batch]['X'].shape
            l = math.sqrt(rho * N)

            m |= process_space(m['X'], l, 'centre_of_mass')
            m |= process_angles(m['A'])

            models[exp] = m

            for stat in stats_names:
                if stat in stats[rho][eta].keys():
                    stats[rho][eta][stat] += m[stat]
                else:
                    stats[rho][eta][stat]  = m[stat]

        # average everything accross all experiments
        for stat in stats_names:
            stats[rho][eta][stat] /= float(count)

        stats[rho][eta]['title'] = m['title']
        stats[rho][eta]['t'] = T
        stats[rho][eta]['l'] = l


    return stats
