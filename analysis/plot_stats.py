#!/usr/bin/python3

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import os

from flock.model import FlockModel
from analysis.stats import autocorrelation, process_space, process_angles
from util.plot import plot_trajectories

from typing import Any, Dict, List, Tuple

titles = {
    'avg_dist_cmass': 'Average distance from centre of mass, ',
    'std_dist_cmass': 'Standard deviation of distance from centre of mass, ',
    'avg_abs_vel': 'Absolute average velocity of particles, ',
    'std_angle': 'Standard deviation of orientation of particles, ',
}


def plots_trajectories(m: FlockModel, stats: Dict[str, Any], pth: str):

    plot_trajectories(m.traj['X'], stats['cmass'], m.l,
        m.string,
        'Trajectories of particles and centre of mass',
        f"{m.title} {m.subtitle}", pth, True, False)

    plot_trajectories(stats['rel_pos'], np.array([[0,0]]* stats['t']), m.l,
        f"{m.string}_relative",
        'Trajectories of particles relative to centre of mass',
        f"{m.title} {m.subtitle}", pth, True, False)


def plots_model_stats(filename: str, subtitle: str, stats: Dict[str, Any], pth: str):

    plt.plot(range(stats['t']), stats['avg_dist_cmass'])
    plt.title('Average distance from centre of mass at time t')
    plt.suptitle(subtitle)
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{E} [||x - x_M||]$ (m)')
    plt.savefig(f"{pth}/avg_dist_cmass_{filename}.png")
    plt.cla()

    plt.plot(range(stats['t']), stats['std_dist_cmass'])
    plt.title('Standard deviation of distance from centre of mass at time t')
    plt.suptitle(subtitle)
    plt.xlabel('t (s)')
    plt.ylabel('$\\sigma_{X_M}$')
    plt.savefig(f"{pth}/std_dist_cmass_{filename}.png")
    plt.cla()

    plt.plot(range(stats['t']), stats['avg_abs_vel'])
    plt.title('Absolute value of average velocity of all particles at time t')
    plt.suptitle(subtitle)
    plt.xlabel('t (s)')
    plt.ylabel('$\sum_i^N \mathbf{v}_{X_i}$')
    plt.savefig(f"{pth}/avg_abs_vel_{filename}.png")
    plt.cla()

    plt.plot(range(stats['t']), stats['std_angle'])
    plt.title('Standard deviation of orientation of all particles at time t')
    plt.suptitle(subtitle)
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{V} [\\theta]$ (rad)')
    plt.savefig(f"{pth}/std_angle_{filename}.png")
    plt.cla()


def plots_autocorrels(filename: str, subtitle: str, stats: Dict[str, Any], window: int, pth: str):

    for val in titles.keys():
        V = stats[val]
        R = autocorrelation(V, window)

        plt.bar(range(1,window+1), R, width=0.3)
        plt.title(titles[val] + 'autocorrelation function')
        plt.suptitle(subtitle)
        plt.xlabel('$\\Delta$ t (s)')
        plt.ylabel('corellation')
        plt.gca().get_xaxis().set_ticks(range(window+1))
        plt.gca().get_yaxis().set_ticks([-1,-.5,0,.5,1])
        plt.savefig(f"{pth}/autocorrel_{val}_{filename}.png")
        plt.cla()



def find_models(path: str, name: str) -> Dict[str, 'FlockModel']:
    """
    Load models from given `path` if they match the `name` parameter and return
    dictionary with directory name as key, and the model object as value
    """
    dirs = [d for d in os.listdir(path)
              if os.path.isdir(os.path.join(path, d)) and name in d ]
    models = { d: FlockModel.load(os.path.join(path, d)) for d in dirs }
    return models


@click.command()
@click.option('--path', default='out/txt/', help='Path to load model data from')
@click.option('--model', default='Vicsek',  help='Model type to load')
def plot_stats(path: str, model: str) -> None:
    """
    After a number of Vicsek simulations were run, plot the results either by
    showing trajectories or averages, variances and autocorrelations

    Run from the root pyflocks/ folder

        python -m analysis.plot_stats [flags]
    """

    models = find_models(path, model)

    plt.rcParams['figure.figsize'] = [12, 8]

    for cur_model in models.keys():
        m = models[cur_model]
        print(f"Plotting model evolution and characteristics to {path}/{cur_model}")

        (t, n, _) = m.traj['X'].shape

        stats   = { 't': t }
        stats |= process_space( m.traj['X'], m.l, m.bounds)
        stats |= process_angles(m.traj['A'], m.params['v'])

        pth = os.path.join(path, m.string)
        title = f"{m.title} {m.subtitle}"

        plots_trajectories(m, stats, pth)
        plots_model_stats(m.string, title, stats, pth)
        plots_autocorrels(m.string, title, stats, 20, pth)


if __name__ == "__main__":
    plot_stats()