#!/usr/bin/python3
from pyflocks.util.plot import prepare_state_plot, savefig, plot_trajectories, plot_state, plot_cmass, plot_sumvec, \
    FlockStyle


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

    plot_state(style, t, sim.traj['X'][t], angles, speeds, sim.l, sim.dt, simple=simple)

    if cmass:
        if traj < 0:
            traj = 20
        plot_cmass(t, sim.traj['X'], sim.l, traj, sim.bounds)
    if sumvec:
        plot_sumvec(t, sim.traj['X'][t], angles, speeds, sim.l, sim.dt, simple=simple)

    savefig(t, sim.title, sim.subtitle, imgpath, simple=simple)
