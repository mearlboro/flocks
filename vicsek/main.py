#!/usr/bin/python
import click

from vicsek.dynamics import VicsekModel
from util.plot import plot_state
from util.util import sim_dir, dump_state

from typing import Any, Dict, List, Tuple



@click.command()
@click.option('-t', default = 100,  help='Number of timesteps')
@click.option('-n', default = 10,   help='Number of particles')
@click.option('-l', default = 5,    help='System size')
@click.option('-e', default = 0.5,  help='Perturbation')
@click.option('-v', default = 0.1,  help='Absolute velocity')
@click.option('-r', default = 1.0,  help='Radius to follow')
@click.option('--bounded', is_flag=True, default=False, help='Bounce against boundaries')
@click.option('--saveimg', is_flag=True, default=False, help='Save images for each state')
def run_vicsek(
        t: int, n: int, l: float, e: float, v: float, r: float,
        bounded: bool, saveimg: bool
    ) -> None:
    """
    Create VicsekModel with params (n, l, e) and run it for T timesteps
    Dump txt file of the state in each step (and image if the flag is set)

    The default values of parameters are as per the original paper:
    https://arxiv.org/abs/cond-mat/0611743
    """

    # initialise model
    sim = VicsekModel(n, l, e, bounded, v, r)

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

        # save current state to image file
        if saveimg:
            print(f'{sim.t}: saving system state to {imgpath}/')
            plot_state(sim.t, sim.X, sim.A, sim.v, l, sim.title, imgpath)
            # bug when saving the first image, so save it again
            if (sim.t == 0):
                plot_state(sim.t, sim.X, sim.A, sim.v, l, sim.title, imgpath)

        sim.update()


if __name__ == "__main__":

    run_vicsek()
