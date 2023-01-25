#!/usr/bin/python
import click
import numpy as np

from flock.vicsek     import VicsekModel
from flock.reynolds   import ReynoldsModel
from flock.kuravicsek import KuramotoVicsekModel
from util.geometry    import EnumBounds, EnumNeighbours
from util.plot        import plot_state, plot_trajectories, FlockStyle, savefig

from typing import Any, Dict, List, Tuple


def __plot(
        sim: 'FlockModel', i: int,
        traj: bool, cmass: bool, sumvec: bool,
        imgpath: str
    ) -> None:

    if type(sim) is KuramotoVicsekModel:
        print(f'{i}: saving system state to {imgpath}/')
        # bug when saving the first image, so save it again
        if (sim.t == 0):
            plot_state(FlockStyle.OSCIL, i, sim.X, sim.P, sim.F, sim.l, sim.dt)
            savefig(i, sim.title, sim.subtitle, imgpath)

        plot_state(FlockStyle.OSCIL, i, sim.X, sim.P, sim.F, sim.l, sim.dt)
    else:
        print(f'{i}: saving system state to {imgpath}/')
        # bug when saving the first image, so save it again
        if (sim.t == 0):
            plot_state(FlockStyle.DOT, i, sim.X, sim.A, sim.V, sim.l, sim.dt)
            savefig(i, sim.title, sim.subtitle, imgpath)

        if traj:
            # remember all positions so far
            Xt[i] = sim.X
            plot_trajectories(i, Xt, sim.l, sim.bounds, cmass)
            style = FlockStyle.DOT
        else:
            style = FlockStyle.ARROW

        plot_state(style, i, sim.X, sim.A, sim.V, sim.l, sim.dt)

    savefig(i, sim.title, sim.subtitle, imgpath)



@click.command()
@click.option('-s',  default = 0,    help='Seed to run simulation: will generate a random seed if not given.')
@click.option('-t',  default = 100,  help='Time to run the simulation')
@click.option('-n',  default = 10,   help='Number of particles')
@click.option('-l',  default = 2.0,  help='System size')
@click.option('-e',  default = 0.4,  help='Perturbation of angular velocity')
@click.option('-v',  default = 0.1,  help='Absolute velocity')
@click.option('-r',  default = 1.0,  help='Radius or number of neighbours to follow')
@click.option('-dt', default = 1.0,  help='Time step')
@click.option('--bounds', required = True,
              type = click.Choice(['PERIODIC', 'REFLECTIVE']),
              help = 'How particles behave at the boundary')
@click.option('--neighbours', required = True,
              type = click.Choice(['METRIC', 'TOPOLOGICAL']),
              help = 'Use neighbours in a radius r or nearest r neighbours')
@click.option('--trajectories', is_flag = True, default = False,
              help = "If true draw particle with trajectories, otherwise velocity vectors")
@click.option('--cmass', is_flag = True, default = False,
              help = "If true draw trajectory of flock centre of mass")
@click.option('--sumvec', is_flag = True, default = False,
              help = "If true draw particle velocity vectors and also their sum vector")
@click.option('--saveimg', is_flag = True, default = False,
              help = 'Save images for each state')
def vicsek(
        s: int, t: int, n: int, l: float, e: float, v: float, r: float, dt: float,
        bounds: str, neighbours: str,
        trajectories: bool, cmass: bool, sumvec: bool,
        saveimg: bool
    ) -> None:
    """
    Create VicsekModel with given params and run it for t timesteps
    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m flock.main vicsek [flags]

    """
    if not s:
        s = np.random.randint(10000)
    # initialise model
    sim = VicsekModel(s, n, l, EnumBounds[bounds], EnumNeighbours[neighbours], e, v, r, dt)

    # initialise folder to save simulation results
    txtpath = sim.mkdir('out/txt')
    if saveimg:
        imgpath = sim.mkdir('out/img')

    Xt = np.zeros((int(t / dt), n, 2))
    # absolute velocity is always the same in Vicsek
    sim.V  = np.ones((n, 1)) * v

    while sim.t < t:
        # save current state to text
        sim.save(txtpath)

        # save current state to image file
        if saveimg:
            i = int(sim.t / sim.dt)
            __plot(sim, i, trajectories, cmass, sumvec, imgpath)

        sim.update()


@click.command()
@click.option('-s',  default = 0,     help='Seed to run simulation: will generate a random seed if not given.')
@click.option('-t',  default = 1,     help='Time to run the simulation')
@click.option('-n',  default = 10,    help='Number of particles')
@click.option('-l',  default = 100.0, help='System size')
@click.option('-a1', default = 0.1,   help='Avoidance')
@click.option('-a2', default = 0.25,  help='Alignment')
@click.option('-a3', default = 0.15,  help='Aggregate')
@click.option('-r',  default = 1.0,   help='Radius or number of neighbours to follow')
@click.option('-dt', default = 0.01,  help='Time step')
@click.option('--bounds', required = True,
              type = click.Choice(['PERIODIC', 'REFLECTIVE']),
              help = 'How particles behave at the boundary')
@click.option('--neighbours', required = True,
              type = click.Choice(['METRIC', 'TOPOLOGICAL']),
              help = 'Use neighbours in a radius r or nearest r neighbours')
@click.option('--trajectories', is_flag = True, default = False,
              help = "If true draw particle with trajectories, otherwise velocity vectors")
@click.option('--sumvec', is_flag = True, default = False,
              help = "If true draw particle velocity vectors and also their sum vector")
@click.option('--saveimg', is_flag = True, default = False,
              help = 'Save images for each state')
def reynolds(
        s: int, t: int, n: int, l: float, a1: float, a2: float, a3: float, r: float, dt: float,
        bounds: str, neighbours: str,
        trajectories: bool, sumvec: bool,
        saveimg: bool
    ) -> None:
    """
    Create Reynolds model with given params and run it for t timesteps
    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m flock.main reynolds [flags]
    """
    if not s:
        s = np.random.randint(10000)
    # initialise model
    sim = ReynoldsModel(s, n, l, EnumBounds[bounds], EnumNeighbours[neighbours],
                        a1, a2, a3, r, dt = dt)

    # initialise folder to save simulation results
    txtpath = sim.mkdir('out/txt')
    if saveimg:
        imgpath = sim.mkdir('out/img')

    Xt = np.zeros((int(t / dt), n, 2))

    while sim.t < t:
        # save current state to text
        sim.save(txtpath)

        # save current state to image file
        if saveimg:
            i = int(sim.t / sim.dt)
            __plot(sim, i, trajectories, False, sumvec, imgpath)

        sim.update()


@click.command()
@click.option('-s',  default = 0,    help='Seed to run simulation: will generate a random seed if not given.')
@click.option('-t',  default = 20,   help='Time to run the simulation')
@click.option('-n',  default = 10,   help='Number of particles')
@click.option('-l',  default = 5.0,  help='System size')
@click.option('-e',  default = 0.5,  help='Perturbation of angular velocity')
@click.option('-v',  default = 0.5,  help='Absolute velocity')
@click.option('-r',  default = 1,    help='Radius or number of neighbours to follow')
@click.option('-k',  default = 0.5,  help='Kuramoto coupling parameter')
@click.option('-f',  default = 1,    help='Intrinsic frequency')
@click.option('-dt', default = 0.1,  help='Time step')
@click.option('--bounds', required = True,
              type = click.Choice(['PERIODIC', 'REFLECTIVE']),
              help = 'How particles behave at the boundary')
@click.option('--neighbours', required = True,
              type = click.Choice(['METRIC', 'TOPOLOGICAL']),
              help = 'Use neighbours in a radius r or nearest r neighbours')
@click.option('--saveimg', is_flag = True, default = False,
              help = 'Save images for each state')
def kuravicsek(
        s: int, t: int, n: int, l: int, e: float, v: float, r: float, k: float, f: float, dt: float,
        bounds: str, neighbours: str, saveimg: bool
    ) -> None:
    """
    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m vicsek.main [flags]

    """
    if not s:
        s = np.random.randint(10000)

    # initialise model
    sim = KuramotoVicsekModel(s, n, l, EnumBounds[bounds], EnumNeighbours[neighbours],
            e, v, r, k, f, dt)

    # initialise folder to save simulation results
    txtpath = sim.mkdir('out/txt')
    if saveimg:
        imgpath = sim.mkdir('out/img')

    while sim.t < t:
        # save current state to text
        sim.save(txtpath)

        # save current state to image file
        if saveimg:
            i = int(sim.t / sim.dt)
            __plot(sim, i, False, False, False, imgpath)

        sim.update()


@click.group()
def options():
    pass

options.add_command(vicsek)
options.add_command(reynolds)
options.add_command(kuravicsek)

if __name__ == "__main__":
    options()
