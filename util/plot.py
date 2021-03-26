#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

from util.geometry import *
from util.util import *

import typing

def prepare_state_plot(l: float) -> None:
    """
    setup plot for an l-sized 2D world with black background
    """
    plt.axis([0,l,0,l])
    plt.style.use("dark_background")
    frame = plt.gca()
    frame.set_aspect("equal")
    frame.axes.get_xaxis().set_ticks(range(l+1))
    frame.axes.get_yaxis().set_ticks(range(l+1))

    return


def plot_vector(X: np.ndarray, a: float, v: float) -> None:
    """
    Plot particle's vector of velocity in its corresponding position with arrow
    at the end using quiver style plot

    Params
    ------
    X
        2D spatial coordinates of point
    a
        angular velocity of particle
    v
        absolute velocity of particle
    """
    ( x,  y) = X
    (vx, vy) = ang_to_vec(a) * v

    # to make them in arrow shape, make headlength and headaxislenght non-zero
    plt.quiver([x], [y], [vx], [vy],
               units='width', angles='xy', scale_units='xy', scale = 1,
               headaxislength=0, headlength = 0, width=.005, color='y')


def plot_oscillator(X: np.ndarray, blink: bool) -> None:
    """
    Plot particle in its corresponding position, and if it's meant to blink,
    then also plot its light, but not its velocity vector

    Params
    ------
    X
        2D spatial coordinates of point
    blink
        is the oscilattor currently blinking
    """
    ( x, y) = X

    if blink:
        plt.scatter(x, y, color='y', marker='o')
    plt.scatter(x, y, color='g', marker='.')


def plot_state_particles(
        t: int, X: np.ndarray, A: np.ndarray,
        v: float, l: int,
        title: str, path: str,
        save: bool = True, show: bool = False
	) -> None:
    """
    Plot the state of a 2D multi-agent/particle system

    Params
    ------
    t
        time unit of the simulation, to be used as filename for generated image
    X
        numpy array of shape (N,2), containing spatial coordinates for N points
    A
        numpy array of shape (N,1), containing angular velocities for N points
    v
        absolute velocity of all particles
    l
        height and width of the system
    title
        title of plot
    path
        path to save file as, should be 'out/img/' folowed by a subdirectory
        named after the model name and parameters
    save
        if True, save images to above path with filename t.jpg
    show
        if True, display the plot
    """
    (n,_) = X.shape

    for i in range(n):
        plot_vector(X[i], A[i], v)

    prepare_state_plot(l)
    plt.xlabel(t)
    plt.title(title)

    if show:
        plt.show()

    if save:
        plt.savefig(f"{path}/{t}.jpg")
        plt.close()

    # clear for next plot
    plt.cla()

    return

def plot_state_oscillators(
        t: int, X: np.ndarray, F: np.ndarray, P: np.ndarray, dt: int, l: int,
        title: str, path: str,
        save: bool = True, show: bool = False
	) -> None:
    """
    Plot the state of a 2D multi-agent/particle system

    Params
    ------
    t
        time unit of the simulation, to be used as filename for generated image
    X
        numpy array of shape (N,2), containing spatial coordinates for N points
    F
        numpy array of shape (N,1), containing the frequency of each oscillator
    P
        numpy array of shape (N,1), containing the phase of each oscillator
    dt
        time increment
    l
        height and width of the system
    title
        title of plot
    path
        path to save file as, should be 'out/img/' folowed by a subdirectory
        named after the model name and parameters
    save
        if True, save images to above path with filename t.jpg
    show
        if True, display the plot
    """
    (n, _) = X.shape

    blinks = [ all(0 <= P[i] <= 1.0 / F[i] / 5) for i in range(len(F)) ]

    for i in range(n):
        plot_oscillator(X[i], blinks[i])

    prepare_state_plot(l)
    plt.xlabel(t)
    plt.title(title)

    if show:
        plt.show()

    if save:
        plt.savefig(f"{path}/{t}.jpg")
        plt.close()

    # clear for next plot
    plt.cla()

    return


def plot_trajectories(
        X: np.ndarray, M: np.ndarray,
        name: str, title: str, suptitle: str, path: str,
        save: bool = True, show: bool = False
    ) -> None:
    """
    Plot the trajectories of all particles and their centre of mass

    Params
    ------
    X
        numpy array of shape (T, N, 2), containing 2D spatial coordinates for N
        points at T time steps
    M
        numpy array of shape (T, 2), containing 2D spatial coordinates for the
        centre of mass at T time steps
    title
        title of plot
    path
        path to save file as, should be 'out/img/' folowed by a subdirectory
        named after the model name and parameters
    save
        if True, save images to above path with filename t.jpg
    show
        if True, display the plot
    """

    (T, N, _) = X.shape

    plt.cla()
    for i in range(N):
        plt.plot(X[:, i, 0], X[:, i, 1], 'k', alpha=0.1)
    plt.plot(M[:, 0], M[:, 1], 'r')

    plt.title(title)
    plt.suptitle(suptitle)

    if show:
        plt.show()

    if save:
        plt.savefig(f"{path}/trajectories_{name}.png")
        plt.close()

    plt.cla()


