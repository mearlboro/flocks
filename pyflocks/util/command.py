import click
from pyflocks.models.factory import FlockFactory
from pyflocks.util.animate import __plot
from pyflocks.util.plot import FlockStyle


@click.group(name='util')
def cli():
    pass


@cli.command()
@click.option('--path', type = str, required = True,
              help = 'Directory with position data')
@click.option('--out', type = str, required = True, default = 'out/img',
              help = 'Directory to output images to')
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
        path: str, out: str, s: int, e: int,
        style: str, simple: bool, color: str,
        traj: int, cmass: bool, sumvec: bool
    ) -> None:
    """
    Read positions and angles, identify model if applicable, and plot each state
    as a PNG image. Directory name gives experimental details e.g.

        experiment_segment_date-time

    or model data e.g.

        {name}_{bounds}_{neighbourhood}(_{paramName}{paramValue})+(_{seed})?

    Run as

        python -m util.animate out/txt/{experiment_or_model_dir} [flags]

    """
    flock = FlockFactory.load(path)
    # TODO: little hack for SL data
    if 'Synch' in flock.string:
        flock.dt = 1/12

    # initialise folder to save images
    imgpath = flock.mkdir(out)

    print(f"Animating {flock.title} in style {style}")

    t = s
    maxt = min(flock.t, e)
    while (t < maxt):
        print(f"{t}: animating system to {imgpath}/")
        __plot(flock, t, FlockStyle[style], traj, cmass, sumvec, imgpath, simple)
        t += 1
