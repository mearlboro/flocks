#!/usr/bin/python
import click
import numpy as np

from util.geometry import EnumBounds, EnumNeighbours
from util.plot import plot_state_vectors, plot_state_particles_trajectories
from util.util import sim_dir, dump_state
from vicsek.dynamics import VicsekModel

from typing import Any, Dict, List, Tuple



@click.command()
@click.option('-t', default = 100,  help='Number of timesteps')
@click.option('-n', default = 10,   help='Number of particles')
@click.option('-l', default = 5,    help='System size')
@click.option('-e', default = 0.4,  help='Perturbation of angular velocity')
@click.option('-v', default = 0.3,  help='Absolute velocity')
@click.option('-r', default = 1.0,  help='Radius or number of neighbours to follow')
@click.option('--trajectories', is_flag = True, default = False,
              help = "If true draw particle with trajectories, otherwise velocity vectors")
@click.option('--bounds', required = True,
              type = click.Choice(['PERIODIC', 'REFLECTIVE']),
              help = 'How particles behave at the boundary')
@click.option('--neighbours', required = True,
              type = click.Choice(['METRIC', 'TOPOLOGICAL']),
              help = 'Use neighbours in a radius r or nearest r neighbours')
@click.option('--saveimg', is_flag = True, default = False,
              help = 'Save images for each state')
def run_simulation(
        t: int, n: int, l: int, e: float, v: float, r: float,
        trajectories: bool, bounds: str, neighbours: str, saveimg: bool
    ) -> None:
    """
    Create VicsekModel with given params and run it for t timesteps
    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m vicsek.main [flags]

    """

    # initialise model
    sim = VicsekModel(n, l, e, EnumBounds[bounds], EnumNeighbours[neighbours], v, r)

    # initialise folder to save simulation results
    txtpath = sim_dir('out/txt', sim.string)
    if saveimg:
        imgpath = sim_dir('out/img', sim.string)

    Xt = np.zeros((t, n, 2))

    while sim.t < t:
        # save current state to text
        print(f'{sim.t}: saving system state to {txtpath}/')
        dump_state(sim.X[:, 0], 'x', txtpath)
        dump_state(sim.X[:, 1], 'y', txtpath)
        dump_state(sim.A[:, 0], 'a', txtpath)

        # save current state to image file
        if saveimg:
            if trajectories:
                print(f'{sim.t}: saving system state to {imgpath}/')
                # remember all positions so far
                Xt[sim.t] = sim.X
                plot_state_particles_trajectories(
                    sim.t, Xt, l, sim.title, imgpath, True)
                # bug when saving the first image, so save it again
                if (sim.t == 0):
                    plot_state_particles_trajectories(
                        sim.t, Xt, l, sim.title, imgpath, True)
            else:
                print(f'{sim.t}: saving system state to {imgpath}/')
                plot_state_vectors(
                    sim.t, sim.X, sim.A, v, l, sim.title, imgpath, True)
                # bug when saving the first image, so save it again
                if (sim.t == 0):
                    plot_state_vectors(
                        sim.t, sim.X, sim.A, v, l, sim.title, imgpath, True)

        sim.update()


if __name__ == "__main__":

    run_simulation()
