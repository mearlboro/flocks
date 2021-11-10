#!/usr/bin/python3

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import os

from util.util import find_models, load_model, aggregate_model_stats
from util.plot import plot_trajectories
from util.analysis import autocorrelation, process_space, process_angles

from typing import Any, Dict, List, Tuple


def plots_trajectories(name: str, m: Dict[str, Any], path: str):
    pth = os.path.join(path, name)
    plot_trajectories(m['X'], m['cmass'],
        name, 'Trajectories of particles and centre of mass',
        m['title'], pth, True, False)

    plot_trajectories(m['rel_pos'], np.array([[0,0]]*m['t']),
        'relative_' + name, 'Trajectories of particles relative to centre of mass',
        m['title'], pth, True, False)


def plots_for_model(name: str, m: Dict[str, Any], path: str):
    pth = os.path.join(path, name)

    print(m['title'])
    plt.plot(range(m['t']), m['avg_dist_cmass'])
    plt.title('Average distance of all particles from centre of mass at time t')
    plt.suptitle(m['title'])
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{E} [||x - x_M||]$ (m)')
    plt.savefig(f"{pth}/avg_dist_cmass_{name}.png")
    plt.cla()

    plt.plot(range(m['t']), m['var_dist_cmass'])
    plt.title('Variance of distance of particles from centre of mass at time t')
    plt.suptitle(m['title'])
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{V} [||x_i - x_M||]$ (m)')
    plt.savefig(f"{pth}/var_dist_cmass_{name}.png")
    plt.cla()

    plt.plot(range(m['t']), m['avg_abs_vel'])
    plt.title('Absolute value of average velocity of all particles at time t')
    plt.suptitle(m['title'])
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{E} [v_a] (m/s)$')
    plt.savefig(f"{pth}/avg_abs_vel_{name}.png")
    plt.cla()

    plt.plot(range(m['t']), m['var_abs_vel'])
    plt.title('Variance of absolute average velocity of all particles at time t')
    plt.suptitle(m['title'])
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{V}[v_a]$ (m/s)')
    plt.savefig(f"{pth}/var_abs_vel_{name}.png")
    plt.cla()

    plt.plot(range(m['t']), m['avg_angle'])
    plt.title('Average orientation of all particles at time t')
    plt.suptitle(m['title'])
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{E} [\\theta]$ (rad)')
    plt.savefig(f"{pth}/avg_angle_{name}.png")
    plt.cla()

    plt.plot(range(m['t']), m['var_angle'])
    plt.title('Variance of orientation of all particles at time t')
    plt.suptitle(m['title'])
    plt.xlabel('t (s)')
    plt.ylabel('$\mathbb{V} [\\theta]$ (rad)')
    plt.savefig(f"{pth}/var_angle_{name}.png")
    plt.cla()


def plots_autocorrelations(name: str, m: Dict[str, Any], window: int, path: str):
    pth = os.path.join(path, name)

    titles = {
        'avg_angle': 'Average orientation of particles, ',
        'avg_abs_vel': 'Absolute average velocity of particles, ',
        'avg_dist_cmass': 'Average distance from centre of mass, ',
    }
    for val in titles.keys():
        V = m[val]
        R = autocorrelation(V, window)

        plt.bar(range(1,window+1), R, width=0.3)
        plt.title(titles[val] + 'autocorrelation function')
        plt.suptitle(m['title'])
        plt.xlabel('$\\Delta$ t (s)')
        plt.ylabel('corellation')
        plt.gca().get_xaxis().set_ticks(range(window+1))
        plt.gca().get_yaxis().set_ticks([-1,-.5,0,.5,1])
        plt.savefig(f"{pth}/autocorrel_{val}_{name}.png")
        plt.cla()


def plot_eta_vs_values(name: str, stats: Dict[str, Any], path: str):

    titles = {
        'avg_angle': 'Average orientation of particles, ',
        'avg_abs_vel': 'Absolute average velocity of particles, ',
        'avg_dist_cmass': 'Average distance from centre of mass, ',
    }
    rhos = stats.keys()

    for rho in rhos:
        etas = [e for e in stats[2.5].keys()]

        for val in titles.keys():
            values = [ np.mean(stats[rho][e][val][:200]) for e in etas ]
            plt.plot(etas, values)
            plt.title(titles[val] + 'vs $\\eta$')
            plt.suptitle(name + f" $\\rho$ = {rho}")
            plt.xlabel('$\\eta$')
            plt.ylabel(titles[val])
            plt.savefig(f"{path}/{name}_rho{rho}_eta_vs_{val}.png")
            plt.cla()


@click.command()
@click.option('--path', default='out/txt/', help='Path to load model data from')
@click.option('--model', default='vicsek',  help='Model type to load')
@click.option('--bounded', is_flag=True,    help='Particles reflect around the edges')
@click.option('--aggregate', is_flag=True,  help='If true, will aggregate simulations with the same params to produce stats, otherwise plot analysis for each simulation')
def plot_results(path: str, model: str, bounded: bool, aggregate: bool) -> None:
    """
    After a number of Vicsek simulations were run, plot the results either by
    showing trajectories or collecting averages and variances, or aggregate
    multiple experiments with the same params

    Run from the root pyflocks/ folder

        python -m vicsek.results [flags]
    """

    models = find_models(path, model)

    # group by bounded and unbounded
    if bounded:
        models = { name:ps for name,ps in models.items() if 'bounded' in name }
        name   = 'Vicsek with reflective boundaries'
    else:
        models = { name:ps for name,ps in models.items() if 'bounded' not in name }
        name   = 'Vicsek model'

    stats = aggregate_model_stats(models, path)

    if aggregate:
        plot_eta_vs_values(name, stats, path)
    else:
        for cur_model in models.keys():
            print("Plotting model evolution and characteristics to {path}/{cur_model}")
            plots_for_model(cur_model, models[cur_model], path)
            plots_trajectories(cur_model, models[cur_model], path)
            plots_autocorrelations(cur_model, models[cur_model], 20, path)


if __name__ == "__main__":
    plot_results()
