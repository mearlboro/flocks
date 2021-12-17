#!/usr/bin/python3

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import os

from flock.model import FlockModel
from analysis.stats import *
from util.plot import plot_trajectories

from typing import Any, Dict, List, Tuple


labels = {
    #'avg_abs_vel': '$\\frac{1}{N v}  \\sum_i^N \mathbf{v}_{X_i}$',
    #'std_angle': '$\\sigma_{\\theta}$',
    'avg_dist_cmass': '$\\mathbb{E}[|X_i, X_M|]$',
    'std_dist_cmass': '$\\sigma_{|X_i, X_M|}$',
}
titles = {
    #'avg_abs_vel': 'Absolute average normalised velocity ',
    #'std_angle': 'Standard deviation of particle direction ',
    'avg_dist_cmass': 'Average distance from centre of mass ',
    'std_dist_cmass': 'Standard deviation of distance from centre of mass ',
}


def plots_trajectories(m: FlockModel, stats: Dict[str, Any], pth: str) -> None:

    plot_trajectories(m.traj['X'], stats['cmass'], m.l,
        m.string,
        'Trajectories of particles and centre of mass',
        f"{m.title} {m.subtitle}", pth, True, False)

    plot_trajectories(stats['rel_pos'], np.array([[0, 0]] * stats['t']), m.l,
        f"{m.string}_relative",
        'Trajectories of particles relative to centre of mass',
        f"{m.title} {m.subtitle}", pth, True, False)


def plots_model_stats(
        filename: str, subtitle: str, stats: Dict[str, Any], pth: str
    ) -> None:

    for stat in titles.keys():
        plt.plot(range(stats['t']), stats[stat])
        plt.title(titles[stat])
        plt.suptitle(subtitle)
        plt.xlabel('t (s)', size = 14)
        plt.ylabel(labels[stat], size = 14)
        plt.savefig(f"{pth}/{filename}_{stat}.png")
        plt.cla()


def plots_autocorrels(
        filename: str, subtitle: str, stats: Dict[str, Any], window: int, pth: str
    ) -> None:

    for stat in titles.keys():
        V = stats[stat]
        R = autocorrelation(V, window)

        plt.bar(range(1, window + 1), R, width = 0.3)
        plt.title(titles[stat] + ', autocorrelation')
        plt.suptitle(subtitle)
        plt.xlabel('$\\Delta$ t (s)', size = 14)
        plt.ylabel('autocorellation', size = 14)
        plt.gca().get_xaxis().set_ticks(range(window + 1))
        plt.gca().get_yaxis().set_ticks([-1, -.5, 0, .5, 1])
        plt.savefig(f"{pth}/{filename}_autocorrel_{stat}.png")
        plt.cla()



def find_models(path: str, name: str) -> Dict[str, 'FlockModel']:
    """
    Load models from given `path` if they match the `name` parameter and return
    dictionary with directory name as key, and the model object as value
    """
    d = os.path.basename(path)
    if name in d:
        return { d: FlockModel.load(path) }

    dirs = [d for d in os.listdir(path)
              if os.path.isdir(os.path.join(path, d)) and name in d ]
    models = { d: FlockModel.load(os.path.join(path, d)) for d in dirs }
    return models


@click.command()
@click.option('--path', default='out/txt/', help='Path to load model data from')
@click.option('--model', default='Vicsek',  help='Model type to load')
@click.option('--out', default='out/plt/', help='Path to save graph to')
def plot_stats(path: str, model: str, out: str) -> None:
    """
    After a Vicsek simulation is run, plot the results either by showing
    trajectories or averages, variances and autocorrelations.

    If an output folder from a simulation is given, analyse just that simulation;
    otherwise, scan for simulation output folders in `path` and analyse all.
    A sub-directory with the same name as the simulation sub-directory will be
    created in `out`.

    Run from the root pyflocks/ folder

        python -m analysis.plot_stats [flags]
    """

    models = find_models(path, model)

    plt.rcParams['figure.figsize'] = [12, 8]

    for cur_model in models.keys():
        m = models[cur_model]
        outpth = os.path.join(out, m.string)
        if not os.path.isdir(outpth):
            os.mkdir(outpth)
        print(f"Plotting model evolution and characteristics to {outpth}")

        (t, n, _) = m.traj['X'].shape

        stats   = { 't': t }
        stats |= process_space( m.traj['X'], m.l, m.bounds)
        #stats |= process_angles(m.traj['A'])

        title = f"{m.title} {m.subtitle}"

        plots_trajectories(m, stats, outpth)
        plots_model_stats(m.string, title, stats, outpth)
        plots_autocorrels(m.string, title, stats, 100, outpth)


if __name__ == "__main__":
    plot_stats()
