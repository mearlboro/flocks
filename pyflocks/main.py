#!/usr/bin/python
import click
import numpy as np
from pyflocks.analysis.command import cli as analysis_cli
from pyflocks.models.kuravicsek import KuramotoVicsekModel
from pyflocks.models.model import FlockModel
from pyflocks.models.reynolds import ReynoldsModel
from pyflocks.models.vicsek import VicsekModel
from pyflocks.util.geometry import EnumBounds, EnumNeighbours
from pyflocks.util.command import cli as util_cli

def __run(sim: 'FlockModel', t: int, out: str) -> None:
    txtpath = sim.mkdir(out)
    while sim.t < t:
        sim.save(txtpath)
        sim.update()

    return


@click.group()
def cli():
    pass


@cli.command()
@click.option('--out', default='out/txt', help='Directory to save trajectories')
@click.option('-s', default=0, help='Seed to run simulation: will generate a random seed if not given.')
@click.option('-t', default=100, help='Time to run the simulation')
@click.option('-n', default=10, help='Number of particles')
@click.option('-l', default=2.0, help='System size')
@click.option('-e', default=0.0, help='Perturbation of angular velocity')
@click.option('-v', default=0.1, help='Absolute velocity')
@click.option('-r', default=1.0, help='Radius or number of neighbours to follow')
@click.option('-dt', default=1.0, help='Time step')
@click.option('--bounds', required=True,
              type=click.Choice(['PERIODIC', 'REFLECTIVE']),
              help='How particles behave at the boundary')
@click.option('--neighbours', required=True,
              type=click.Choice(['METRIC', 'TOPOLOGICAL']),
              help='Use neighbours in a radius r or nearest r neighbours')
def vicsek(out: str,
           s: int, t: int, n: int, l: float,
           e: float, v: float, r: float,
           dt: float, bounds: str, neighbours: str
           ) -> None:
    """
    Create VicsekModel with given params and run it for t timesteps.

    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m flock.main vicsek [flags]

    """
    if not s:
        s = np.random.randint(10000)
    sim = VicsekModel(s, n, l, EnumBounds[bounds], EnumNeighbours[neighbours], dt,
                      params={'eta': e, 'v': v, 'r': r})
    __run(sim, t, out)


@cli.command()
@click.option('--out', default='out/txt', help='Directory to save trajectories')
@click.option('-s', default=0, help='Seed to run simulation: will generate a random seed if not given.')
@click.option('-t', default=10, help='Time to run the simulation')
@click.option('-n', default=10, help='Number of particles')
@click.option('-l', default=100.0, help='System size')
@click.option('-a1', default=0.15, help='Aggregate')
@click.option('-a2', default=0.05, help='Avoidance')
@click.option('-a3', default=0.25, help='Alignment')
@click.option('-r', default=1.0, help='Radius or number of neighbours to follow')
@click.option('-dt', default=0.1, help='Time step')
@click.option('--bounds', required=True,
              type=click.Choice(['PERIODIC', 'REFLECTIVE']),
              help='How particles behave at the boundary')
@click.option('--neighbours', required=True,
              type=click.Choice(['METRIC', 'TOPOLOGICAL']),
              help='Use neighbours in a radius r or nearest r neighbours')
def reynolds(out: str,
             s: int, t: int, n: int, l: float,
             a1: float, a2: float, a3: float, r: float,
             dt: float, bounds: str, neighbours: str
             ) -> None:
    """
    Create Reynolds model with given params and run it for t timesteps.

    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m flock.main reynolds [flags]
    """
    if not s:
        s = np.random.randint(10000)
    sim = ReynoldsModel(s, n, l, EnumBounds[bounds], EnumNeighbours[neighbours], dt,
                        params={'aggregate': a1, 'avoidance': a2, 'alignment': a3, 'r': r})
    __run(sim, t, out)


@cli.command()
@click.option('--out', default='out/txt', help='Directory to save trajectories')
@click.option('-s', default=0, help='Seed to run simulation: will generate a random seed if not given.')
@click.option('-t', default=10, help='Time to run the simulation')
@click.option('-n', default=10, help='Number of particles')
@click.option('-l', default=5.0, help='System size')
@click.option('-e', default=0.5, help='Perturbation of angular velocity')
@click.option('-v', default=0.3, help='Absolute velocity')
@click.option('-r', default=1, help='Radius or number of neighbours to follow')
@click.option('-k', default=0.5, help='Kuramoto coupling parameter')
@click.option('-f', default=1, help='Intrinsic frequency')
@click.option('-dt', default=0.1, help='Time step')
@click.option('--bounds', required=True,
              type=click.Choice(['PERIODIC', 'REFLECTIVE']),
              help='How particles behave at the boundary')
@click.option('--neighbours', required=True,
              type=click.Choice(['METRIC', 'TOPOLOGICAL']),
              help='Use neighbours in a radius r or nearest r neighbours')
def kuravicsek(out: str,
               s: int, t: int, n: int, l: int,
               e: float, v: float, r: float, k: float, f: float,
               dt: float, bounds: str, neighbours: str
               ) -> None:
    """
    Create Kuramoto-Vicsek model with given params and run it for t timesteps.

    Dump txt file of the state in each step (and image if the flag is set)

    Run from the root pyflocks/ folder

        python -m flocks.main kuravicsek [flags]
    """
    if not s:
        s = np.random.randint(10000)
    sim = KuramotoVicsekModel(s, n, l, EnumBounds[bounds], EnumNeighbours[neighbours], dt,
                              params={'eta': e, 'v': v, 'r': r, 'k': k, 'f': f})
    __run(sim, t, out)


cli.add_command(analysis_cli)
cli.add_command(util_cli)

if __name__ == "__main__":
    cli()
