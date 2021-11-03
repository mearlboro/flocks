#!/usr/bin/python
import click

from vicsek.dynamics import KuramotoFlock
from util.geometry import EnumBounds, EnumNeighbours
from util.plot import plot_state_oscillators
from util.util import sim_dir, dump_state

from typing import Any, Dict, List, Tuple



@click.command()
@click.option('-t', default = 100,  help='Number of timesteps')
@click.option('-n', default = 10,   help='Number of particles')
@click.option('-l', default = 5,    help='System size')
@click.option('-e', default = 0.5,  help='Perturbation of angular velocity')
@click.option('-k', default = 0.5,  help='Kuramoto coupling parameter')
@click.option('-r', default = 1,    help='Radius or number of neighbours to follow')
@click.option('-f', default = 1,    help='Intrinsic frequency')
@click.option('--bounds', required = True,
              type = click.Choice(['PERIODIC', 'REFLECTIVE']),
              help = 'How particles behave at the boundary')
@click.option('--neighbours', required = True,
              type = click.Choice(['METRIC', 'TOPOLOGICAL']),
              help = 'Use neighbours in a radius r or nearest r neighbours')
@click.option('--saveimg', is_flag = True, default = False,
              help = 'Save images for each state')
def run_simulation(
        t: int, n: int, l: int, e: float, k: float, r: float, f: float,
        bounds: str, neighbours: str, saveimg: bool
    ) -> None:
    """
    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m vicsek.main [flags]

    """

    # initialise model with some fixed params
    sim = KuramotoFlock(n, l, e, k, EnumBounds[bounds], EnumNeighbours[neighbours], r = r, f = f)

    # initialise folder to save simulation results
    txtpath = sim_dir('out/txt', sim.string)
    if saveimg:
        imgpath = sim_dir('out/img', sim.string)

    while sim.t < t:
        # save current state to text
        print(f'{sim.t}: saving system state to {txtpath}/')
        dump_state(sim.X[:, 0], 'x', txtpath)
        dump_state(sim.X[:, 1], 'y', txtpath)
        dump_state(sim.A[:, 0], 'a', txtpath)
        dump_state(sim.P[:, 0], 'p', txtpath)

        # save current state to image file
        if saveimg:
            print(f'{sim.t}: saving system state to {imgpath}/')
            plot_state_oscillators(
                int(sim.t / sim.dt), sim.X, sim.F, sim.P, sim.dt, sim.l, sim.title, imgpath)
            # bug when saving the first image, so save it again
            if (sim.t == 0):
                plot_state_oscillators(
                    int(sim.t / sim.dt), sim.X, sim.F, sim.P, sim.dt, sim.l, sim.title, imgpath)

        sim.update()


if __name__ == "__main__":

    run_simulation()
