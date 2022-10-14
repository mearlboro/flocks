#!/usr/bin/python3

import click
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import numpy.random as rn
import os

from flock.model import FlockModel
from analysis import order
from util.plot import plot_trajectories, plot_state_particles_trajectories

from typing import Any, Dict, List, Tuple

labels = {
    'vicsek_order'      : '$v_a(t)$',
    'mean_angle'        : '$\\tilde{\\theta}(t)$',
    'var_angle'         : '$\\sigma^2_{\\theta}(t)$',
    'mean_dist_cmass'   : '$\\tilde{d}_{\\tilde{X}}(t)$',
    'var_dist_cmass'    : '$\\sigma^2_{\\tilde{d}}(t)$',
    'mean_neighbours'   : '$\\tilde{\\rho}(t)$',
    'mean_dist_nearest' : '$\\tilde{\\delta}(t)$'
}
titles = {
    'vicsek_order'      : 'Vicsek order parameter',
    'mean_angle'        : 'Mean particle direction',
    'var_angle'         : 'Spread of particle direction',
    'mean_dist_cmass'   : 'Mean distance from centre',
    'var_dist_cmass'    : 'Spread from centre',
    'mean_neighbours'   : 'Mean number of interaction neighbours',
    'mean_dist_nearest' : 'Mean distance to nearest neighbour'
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


def plot_order_params(m: FlockModel, out: str = 'out/plt/') -> None:
    outpth = os.path.join(out, m.string)
    if not os.path.isdir(outpth):
        os.mkdir(outpth)
    print(f"Plotting model order params to {outpth}")

    (t, n, _) = m.traj['X'].shape

    Xt = m.traj['X']
    At = m.traj['A']

    params = order.param(Xt, At, m.l, m.params['r'], m.bounds)

    # plot psis
    plt.rcParams['figure.figsize'] = 7,5
    plt.xlabel('t (s)')
    plt.ylabel('$\\Psi^{(1,1)}_{\\tilde{x}}(t)$',)
    plt.plot(range(n, t), params['psi_cmass_loc'], label = 'Local MI')
    plt.plot(range(n, t), params['psi_cmass_loc2'], label = 'Local MI (median filtered)')
    plt.plot(range(n, t), params['psi_cmass_avg'], label = 'Average MI')
    plt.title('Emergence $\\Psi$ for positions and centre of mass')
    plt.legend()
    plt.savefig(f"{outpth}/{m.string}_psi_cmass_2.png")
    plt.close('all')

    # plot order params
    plt.rcParams['figure.figsize'] = 4.5,3
    for i,p  in enumerate(titles):
        print(f"Plotting {p} for model")
        plt.plot(range(t), params[p])
        plt.title(titles[p])
        plt.xlabel('t (s)')
        plt.ylabel(labels[p])
        plt.tight_layout()
        plt.savefig(f"{outpth}/{m.string}_{p}.png")
        plt.cla()
    plt.close('all')

    # investigate first, last, low and high psi states
    if t > 100:
        ts = [0, 50, 100, t-1] + sorted(list(params['psi_cmass_minmax'].keys()))
    else:
        ts = [0, 10, 50,  t-1] + sorted(list(params['psi_cmass_minmax'].keys()))

    # plot histogram of directions
    plt.rcParams['figure.figsize'] = 10,5
    print("Plotting histograms of direction")
    fig = plt.figure()
    for i, time in enumerate(ts):
        ax = plt.subplot(241 + i)
        for a in m.traj['A'][time]:
            plt.scatter(np.cos(a), np.sin(a))
        plt.title(f't={time}')
        ax.set(xlim=[-1.1, 1.1], ylim=[-1.1, 1.1], aspect=1)
    plt.tight_layout()
    plt.draw()
    plt.savefig(f"{outpth}/{m.string}_directions.png")
    plt.close('all')

    # plot histogram of positions
    print("Plotting histograms of position")
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 4), axes_pad=0.4,
                     cbar_mode="single", cbar_location="right", cbar_pad=0.2)
    for i, time in enumerate(ts):
        X = m.traj['X'][time]
        h = grid[i].hist2d(X[:, 0], X[:, 1], bins=range(0, int(m.l)+1), cmap='YlOrRd')
        grid[i].set(xlim=[0, m.l], ylim=[0, m.l], aspect=1)
        if i >= 4:
            grid[i].set_title(f"t={time} $\\Psi$={round(params['psi_cmass_loc'][time-n], 2)}")
        else:
            grid[i].set_title(f"t={time}")
    fig.colorbar(h[3], cax=grid.cbar_axes[0], orientation='vertical')
    plt.draw()
    plt.savefig(f"{outpth}/{m.string}_positions.png")
    plt.close('all')

    # and plot the trajectories as well for the relevant timepoints
    print("Plotting interesting states with trajectories and centre of mass")
    # plot the first one twice to avoid a bug
    for i, time in enumerate([ts[4]]+ts[4:]):
        Xt = m.traj['X'][:time+1]
        plot_state_particles_trajectories(time, Xt, m.l, m.bounds, m.title,
            m.subtitle, outpth, cmass=True)
    plt.close('all')



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
    trajectories or averages, variances, order parameters, and histograms
    for the most 'interesting' states.

    If an output folder from a simulation is given, analyse just that simulation;
    otherwise, scan for simulation output folders in `path` and analyse all.
    A sub-directory with the same name as the simulation sub-directory will be
    created in `out`.

    Run from the root pyflocks/ folder

        python -m analysis.plot_stats [flags]
    """

    models = find_models(path, model)

    plt.rcParams['figure.figsize'] = [12, 8]

    for cur_model in models.values():
        plot_order_params(cur_model, out)

if __name__ == "__main__":
    plot_stats()
