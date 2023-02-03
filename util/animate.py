#!/usr/bin/python3
import click
import numpy as np

from flock.factory import FlockFactory
from util.geometry import EnumBounds, EnumNeighbours
from util.plot     import *

from typing import Any, Dict, List, Tuple


def __plot(
        sim: 'FlockModel', t: int,
        style: 'FlockStyle', traj: int, cmass: bool, sumvec: bool,
        imgpath: str, simple: bool
    ) -> None:

    # bug when saving the first image, so save it again
    if (t == 0):
        prepare_state_plot(sim.l)
        savefig(0, sim.title, sim.subtitle, imgpath)

    coords = sim.traj['X'][t]
    angles = sim.traj['A'][t]
    # absolute velocity may not change in all models, so there is no trajectory
    if 'V' in sim.traj.keys():
        speeds = sim.traj['V'][t]
    else:
        speeds = sim.V

    # plot phase and frequency of oscillator instead of angle and speed
    if 'Kuramoto' in str(type(sim)) and style == FlockStyle.OSCIL:
        angles = sim.traj['P'][t]
        # absolute velocity may not change in all models, so there is no trajectory
        if 'F' in sim.traj.keys():
            speeds = sim.traj['F'][t]

    if traj > 0:
        plot_trajectories(t, sim.traj['X'], sim.l, traj, sim.bounds)

    plot_state(style, t, sim.traj['X'][t], angles, speeds, sim.l, sim.dt, simple = simple)

    if cmass:
        if traj < 0:
            traj = 20
        plot_cmass(t, sim.traj['X'], sim.l, traj, sim.bounds)
    if sumvec:
        plot_sumvec(t, sim.traj['X'][t], angles, speeds, sim.l, sim.dt, simple = simple)

    savefig(t, sim.title, sim.subtitle, imgpath, simple = simple)



@click.command()
@click.option('--path', type = str, required = True,
              help = 'Directory with position data')
@click.option('-s', default = 0,
              help = 'Time increment to start plotting from')
@click.option('-e', default = 1000000,
              help = 'Time increment to plot until')
@click.option('--style', type = click.Choice(FlockStyle.names()),
              help = 'Style in which to plot each player', default = 'DOT')
@click.option('--simple', is_flag = True,
              help = "If set, don't include details in plot title", default = False)
@click.option('--color', default = 'w',
              help = 'Colour of players, random if unset')
@click.option('--traj', default = 0, type = int,
              help = 'Length of trajectories, 0 for none')
@click.option('--cmass', is_flag = True,
              help = 'Draw the centre of mass trajectory', default = False)
@click.option('--sumvec', is_flag = True,
              help = 'Draw the sum vector of all velocities', default = False)
def plot(
        path: str, s: int, e: int,
        style: str, simple: bool, color: str,
        traj: int, cmass: bool, sumvec: bool
    ) -> None:
    """
    Read positions and angles, identify model if applicable, and plot each state
    as a PNG image. Directory name gives experimental details e.g.

        experiment_segment_date-time

    or model data e.g.

        {modelName}_{bounds}_{neighbourhood}(_{paramName}{paramValue})+(_{seed})?

        python -m flock.animate plot out/txt/experiment1 [flags]

    """
    flock = FlockFactory.load(path)
    # TODO: little hack for SL data
    if 'Synch' in flock.string:
        flock.dt = 1/12

    # initialise folder to save images
    imgpath = flock.mkdir('out/img')

    print(f"Animating {flock.title} in style {style}")

    t = s
    maxt = min(flock.t, e)
    while (t < maxt):
        print(f"{t}: animating system to {imgpath}/")
        __plot(flock, t, FlockStyle[style], traj, cmass, sumvec, imgpath, simple)
        t += 1


if __name__ == "__main__":
    plot()
