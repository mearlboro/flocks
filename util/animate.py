#!/usr/bin/python3
import click
import numpy as np

from flock.model   import Flock, FlockModelTypes
from util.geometry import EnumBounds, EnumNeighbours
from util.plot     import *

from typing import Any, Dict, List, Tuple


@click.command()
@click.option('-i', type = str, required = True,
              help = 'Directory with position data')
@click.option('-s', default = 0,
              help = 'Time increment to start plotting from')
@click.option('-e', default = 1000000,
              help = 'Time increment to plot until')
@click.option('--style', type = click.Choice([ 'arrow', 'dot', 'line', 'oscil' ]),
              help = 'Style in which to plot each player', default = 'dot')
@click.option('--color', default = 'w',
              help = 'Colour of players, random if unset')
@click.option('--traj', default = 0,
              help = 'Length of trajectories, 0 for none')
@click.option('--order', type = click.Choice([ 'none', 'sumvec', 'cmass' ]),
              help = 'Draw the sum vector of all velocities, or the centre of mass')
def plot(
        i: str, s: int, e: int, style: str, color: str, traj: bool, order: str
    ) -> None:
    """
    Read positions and angles, identify model if applicable, and plot each state
    as a PNG image. Directory name gives experimental details e.g.

        experiment_segment_date-time

    or model data e.g.

        {modelName}_{bounds}_{neighbourhood}(_{paramName}{paramValue})+(_{seed})?

        python -m flock.animate plot out/txt/experiment1 [flags]

    """
    flock = None

    # get the experiment details from folder name
    if any(m.name.lower() in i.lower() for m in FlockModelTypes):
        # TODO plot actual simulations
        pass
    else:
        name, segment = i.split('_')[:2]
        if 'SynchLive' in name:
            flock = Flock.load(i, 1/12)

    # initialise folder to save images
    imgpath = flock.mkdir('out/img')

    t = s
    maxt = min(flock.t, e)

    while (t < maxt):
        print(f'{t}: animating system to {imgpath}/')
        X = flock.traj['X'][t]
        A = flock.traj['A'][t]
        V = flock.traj['V'][t]

        # bug when saving the first image, so save it twice
        if (t == s):
            plot_state(FlockStyle.fromStr(style), t, X, A, V, flock.l, flock.dt, color, True)
            savefig(t, '', '', imgpath, simple = True)

        if traj:
            plot_trajectories(t - s, flock.traj['X'][s:], traj, flock.l, flock.bounds, order == 'cmass')

        plot_state(FlockStyle.fromStr(style), t, X, A, V, flock.l, flock.dt, color, True)
        savefig(t, '', '', imgpath, simple = True)

        t += 1


@click.group()
def options():
    pass

options.add_command(plot)

if __name__ == "__main__":
    options()
