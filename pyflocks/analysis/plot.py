import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os

from pyflocks.models.model import Flock

from typing import Any, Dict, List


def order_params(
        m: 'Flock', params: Dict['EnumParams', Any], out: str = 'out/plt/'
    ) -> None:
    """
    Given a Flock object, that may contain either simulated or experimental data
    as numpy arrays of trajectories and angles, and the instantaneous order params
    computed at each timestep, plot how they evolve over time for the system.
    """
    outpth = os.path.join(out, m.string)
    if not os.path.isdir(outpth):
        os.mkdir(outpth)
    print(f"Plotting order params to {outpth}")

    # plot order params
    plt.rcParams['figure.figsize'] = 6,4
    for i,p  in enumerate(params.keys()):
        print(f"Plotting {p} for experiment or simulation")
        plt.plot(np.arange(0, m.t) / (1 / m.dt), params[p])
        plt.title(p.title())
        plt.xlabel('t (s)')
        plt.ylabel(p.label())
        plt.tight_layout()
        plt.savefig(f"{outpth}/{p}.png", dpi = 320)
        plt.cla()
    plt.close('all')


def states(
        states: List[int], m: 'Flock', out: str = 'out/plt/'
    ) -> None:
    """
    Given a Flock object, that may contain either simulated or experimental data
    as numpy arrays of trajectories and angles, plot histograms of position and
    angles for given states.
    """
    outpth = os.path.join(out, m.string)
    if not os.path.isdir(outpth):
        os.mkdir(outpth)
    print(f"Plotting histograms for interesting states to {outpth}")

    (t, n, _) = m.traj['X'].shape
    Xt = m.traj['X']
    At = m.traj['A']
    l  = m.l

    # plot histogram of directions
    plt.rcParams['figure.figsize'] = 10, 5
    print("Plotting histograms of direction")
    fig = plt.figure()
    for i, time in enumerate(states):
        ax = plt.subplot(241 + i)
        for a in At[time]:
            plt.scatter(np.cos(a), np.sin(a))
        plt.title(f't={time}')
        ax.set(xlim=[-1.1, 1.1], ylim=[-1.1, 1.1], aspect=1)
    plt.tight_layout()
    plt.draw()
    plt.savefig(f"{outpth}/hist_dir.png", dpi = 320)
    plt.close('all')

    # plot histogram of positions
    print("Plotting histograms of position")
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 4), axes_pad=0.4,
                     cbar_mode="single", cbar_location="right", cbar_pad=0.2)
    for i, time in enumerate(states):
        X = Xt[time]
        h = grid[i].hist2d(X[:, 0], X[:, 1], bins=range(0, int(l)+1), cmap='YlOrRd')
        grid[i].set(xlim=[0, l], ylim=[0, l], aspect=1)
        grid[i].set_title(f"t={time}")
    fig.colorbar(h[3], cax=grid.cbar_axes[0], orientation='vertical')
    plt.draw()
    plt.savefig(f"{outpth}/hist_pos.png", dpi = 320)
    plt.close('all')


markers = itertools.cycle(['.', 'o', 'v', '^', '>', '<', '8', 's', 'p', 'P', '*',
    'h', 'X', 'D', 'd' ])

def aggregate_2param(
        filename: str, stats: Dict[float, Dict[float, Any]],
        conp: List[str], ordp: 'EnumParams',
        title: str, path: str = 'out/plt/'
    ):
    """
    Plot aggregated statistics or order parameters grouping by 2 model parameters.
    """
    if len(conp) != 2:
        raise ValueError(f'Can only plot aggregate plot with 2 params, but {conp} given')
        exit(0)

    plt.rcParams['figure.figsize'] = [7,5]
    for p0 in stats.keys():
        xval = [ p1 for p1 in sorted(stats[p0].keys()) ]
        yval = [ stats[p0][p1][f'{ordp}_mean'] for p1 in xval ]
        yerr = [ stats[p0][p1][f'{ordp}_std']  for p1 in xval ]
        plt.errorbar(xval, yval, yerr = np.sqrt(yerr), label = ordp.label(),
            linestyle = '--', marker = next(markers), c = np.random.rand(3,))

        if 1 < len(conp[0]) < 4:
            tlab = f"$\\{conp[0]}$"
        else:
            tlab = f"${conp[0]}$"
        if 1 < len(conp[1]) < 4:
            xlab = f"$\\{conp[1]}$"
        else:
            xlab = f"${conp[1]}$"
        plt.title(ordp.title() + f" vs {xlab}", fontsize = 12)
        plt.suptitle(title + f" ({tlab} = {p0})", fontsize = 13)
        plt.ylabel(ordp.label(), fontsize = 12)
        plt.xlabel(xlab, fontsize = 12)

        plt.tight_layout()
        plt.savefig(f"{path}/{filename}_{conp[0]}{p0}_{conp[1]}_vs_{ordp}.png", dpi = 320)
        plt.cla()


def aggregate_3param(
        filename: str, stats: Dict[float, Dict[float, Any]],
        conp: List[str], ordp: 'EnumParams',
        title: str, path: str = 'out/plt/'
    ):
    """
    Plot aggregated statistics or order parameters grouping by 3 model parameters.
    """

    if len(conp) != 3:
        raise ValueError(f'Can only plot aggregate plot with 3 params, but {conp} given')
        exit(0)
    import pdb; pdb.set_trace()

    plt.rcParams['figure.figsize'] = [7,5]
    for p0 in stats.keys():
        xval = [ p1 for p1 in stats[p0].keys() ]

        groups = list(set([ p2 for p1 in xval for p2 in stats[p0][p1].keys() ]))
        for p2 in groups:
            yval = [ np.mean(stats[p0][p1][p2][ordp]) for p1 in xval
                                                      if p2 in stats[p0][p1].keys()]

            plt.plot(xval, yval, label = f'{conp[2]} = {p2}',
                linestyle = '--', marker = next(markers), c = np.random.rand(3,))

        if 1 < len(conp[1]) < 4:
            xlab = f"$\\{conp[1]}$"
        else:
            xlab = f"${conp[1]}$"
        plt.title(ordp.title() + f" vs {xlab}$", fontsize = 14)
        plt.suptitle(title + f" (${conp[0]}$ = {p0})", fontsize = 16)
        plt.ylabel(ordp.label(), fontsize = 14)
        plt.xlabel(xlab, fontsize = 14)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{path}/{filename}_{conp[0]}{p0}_{conp[1]}_{conp[2]}_vs_{ordp}.png", dpi = 320)
        plt.cla()

