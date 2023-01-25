#!/usr/bin/python3
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

from util.geometry import *

import typing


class FlockStyle(Enum):
    ARROW = 0
    DOT   = 1
    LINE  = 2
    OSCIL = 3

    @classmethod
    def fromStr(self, name: str) -> 'FlockStyle':
        if name.lower() == 'arrow':
            return self.ARROW
        if name.lower() == 'dot':
            return self.DOT
        if name.lower() == 'line':
            return self.LINE
        if name.lower() == 'oscil':
            return self.OSCIL


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Prepare background
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def prepare_state_plot(l: float, ticks: bool = True) -> None:
    """
    setup plot for an l-sized 2D world with black background
    """
    plt.rcParams['figure.figsize'] = 7,5
    plt.axis([0,l,0,l])
    plt.style.use("dark_background")
    frame = plt.gca()
    frame.set_aspect("equal")
    if ticks:
        if l < 20:
            frame.axes.get_xaxis().set_ticks(range(int(l)+1))
            frame.axes.get_yaxis().set_ticks(range(int(l)+1))
    else:
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
    return


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Plot individual particle in each style
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def plot_dot(X: np.ndarray, col: str = 'w') -> None:
    """
    Plot single particle/agent as dot

    Params
    ------
    X
        numpy array of shape (2,) containing 2D coordinates
    """
    (x,  y) = X
    plt.scatter(x, y, color = col, marker = '.')
    return


def plot_vector(
        X: np.ndarray, a: float, v: float,
        V: np.ndarray = None, col: str = 'w', quiver: bool = True
    ) -> None:
    """
    Plot particle's vector of velocity in its corresponding position with arrow
    at the end using quiver style plot. Use position, angle and absolute velocity,
    or 2D direction vector if specified.

    Params
    ------
    X
        2D spatial coordinates of point
    a
        angular velocity of particle
    v
        absolute velocity of particle
    V
        alternatively, 2D vector for the velocity of particle
    """
    ( x,  y) = X
    if V:
        (vx, vy) = V
    else:
        (vx, vy) = ang_to_vec(a) * v

    # to make them in arrow shape, make headlength and headaxislenght non-zero
    if quiver:
        quivsize = 2
    else:
        quivsize = 0

    plt.quiver([x], [y], [vx], [vy],
               units = 'width', angles = 'xy', scale_units = 'xy', scale  =  0.5,
               headaxislength = quivsize, headlength = quivsize, width = .01, color = col)
    return


def plot_oscillator(
        X: np.ndarray, p: float, f: float, dt: float, blink: bool,
        col1: str = 'y', col2: str = 'g'
    ) -> None:
    """
    Plot particle in its corresponding position, and if it's meant to blink,
    then also plot its light, but not its velocity vector.

    Params
    ------
    X
        2D spatial coordinates of point
    p
        the current oscillator's phase / angle (in radians)
    f
        the current oscillator's frequency / anguar speed (in Hz)
    dt
        time increment (in seconds)
    blink
        if set, then only blink once every rotation, otherwise fade a light
        in and out to show oscillator behavious
    """
    (x, y) = X

    if blink:
        # blink only once every rotation for one frame as phase just got >0
        if 0 <= p <= 2 * np.pi * f * dt:
            plt.scatter(x, y, color = 'y', marker = 'o')
    else:
        # fade a light in and out, such that it's off when phase is 0 but fully
        # bright when phase is pi
        if p > np.pi:
            p = 2* np.pi - p
        elif p < 0:
            p = abs(p)
        p /= np.pi
        plt.scatter(x, y, color = col1, marker = 'o', alpha = p)

    plt.scatter(x, y, color = col2, marker = '.')
    return


def plot_trajectory(
        t: int, Xit: np.ndarray, l: float,
        col: str = 'grey', ts: int = 100
    ) -> None:
    """
    Plot a particle's trajectory for the last ts timepoints

    Params
    ------
    t
        time unit of the simulation
    Xit
        numpy array of shape (t, 2) with all the points previous positions
    l
        size of space
    col
        colour of trajectory (matplotlib colour name)
    ts
        plot last ts timepoints
    """
    if t > ts:
        Xit = Xit[t-ts:t+1]
    else:
        Xit = Xit[0:t+1]

    if t > 0:
        # to avoid cross lines for periodic boundaries use a masked array
        abs_Xt   = np.abs(np.diff(Xt[:, 0]))
        mask     = np.hstack([ abs_Xt >= l-1, [False]])
        mask_Xt0 = np.ma.MaskedArray(Xt[:, 0], mask)

        abs_Xt   = np.abs(np.diff(Xt[:, 1]))
        mask     = np.hstack([ abs_Xt >= l-1, [False]])
        mask_Xt1 = np.ma.MaskedArray(Xt[:, 1], mask)

        if col == 'grey':
            alpha = 0.3
        else:
            alpha = 1
        plt.plot(mask_Xt0, mask_Xt1, col, alpha=alpha)
    return



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Plot state of all particle at time t with given style
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def plot_trajectories(
        t: int, Xt: np.ndarray, ts: int, l: float,
        topology: EnumBounds = EnumBounds.REFLECTIVE,
        cmass: bool = False,
    ) -> None:
    """
    Plot the trajectories ts discrete timesteps before time t. To be used before
    plot_state.

    Params
    ------
    t
        final time unit of the simulation
    Xt
        numpy array of shape (T,N,2), containing spatial coordinates for N
        particles at all timepoints
    ts
        number of time steps shown in the trajectory, trajectories will be shown
        from times t-ts to t.
    l
        height and width of the system
    topology
        whether the space's boundaries are reflective or periodic
    cmass
        whether to plot the trajectory of the centre of mass
    """
    (_, n, _) = Xt.shape

    prepare_state_plot(l)
    for i in range(n):
        plot_trajectory(t, Xt[:, i], l, ts = ts)

    if cmass:
        M = np.array([ centre_of_mass(X, l, topology) for X in Xt ])
        plot_trajectory(t, M, l, 'yellow', ts)
    return


def plot_state(
        style: FlockStyle,
        t: int, X: np.ndarray, A: np.ndarray, V: np.ndarray, l: int, dt: float,
        col: str = 'w', simple: bool = False
    ) -> None:
    """
    Plot the state of a 2D multi-agent/particle system

    Params
    ------
    style
        what style to plot in: dot, line, arrow, oscillator
    t
        time unit of the simulation, to be used as filename for generated image
    X
        np array of shape (N, 2), containing the spatial coordinates for N points
    A
        np array of shape (N, 1), containing angle of velocity for N points at
        time t, or phase if it's an oscillator
    V
        np array of shape (N, d), containing some form of velocity, either speed
        in shape (N, 1), velocity vecors of shape (N, 2), or frequencies for
        oscillators of shape (N, 1)
    l
        height and width of the system
    dt
        time increment
    col
        colour of dots
    simple
        if set, don't show ticks, subtitle, labels, annotations
    """
    (n, _) = X.shape
    vdim = V.shape[1] if len(V.shape) == 2 else 1

    prepare_state_plot(l, ticks = not simple)

    if style == FlockStyle.ARROW:
        for i in range(n):
            if vdim == 1:
                plot_vector(X[i], A[i], V[i], None, col)
            else:
                plot_vector(X[i], A[i], None, V[i], col)
    elif style == FlockStyle.DOT:
        for i in range(n):
            plot_dot(X[i], col)
    elif style == FlockStyle.LINE:
        for i in range(n):
            if vdim == 1:
                plot_vector(X[i], A[i], V[i], None, col, quiver = False)
            else:
                plot_vector(X[i], A[i], None, V[i], col, quiver = False)
    elif style == FlockStyle.OSCIL:
        for i in range(n):
            plot_oscillator(X[i], p = A[i], f = V[i], dt = dt, blink = False)
    else:
        raise ValueError(f"Style {style} not supported by `plot_state`.")

    plt.xlabel(t)

    return



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
plot observable parameters on top of the system state
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#def plot_order(
#        param: EnumOrder
#    )
#
#    if cmass:
#        M = np.array([ centre_of_mass(X, l, topology) for X in Xt ])
#        print(M.shape)
#        plot_trajectory(t, M, l, 'yellow')
#
#    if sumvec:
#        S = sum_vec_ang(A, V) / n
#        plt.plot([l/2, S[0] + l/2], [l/2, S[1] + l/2], 'yellow', linewidth=3)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Save or show the state plot
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def savefig(
        t: int, title: str, subtitle: str, path: str, simple: bool = False,
        save: bool = True, show: bool = False, clear: bool = True
    ) -> None:
    """
    Title, label, annotate and save current figure.

    Params
    ------
    t
        time unit of the simulation, to be used as filename for generated image
    title
        title of plot
    subtitle
        subtitle of plot
    path
        path to save file as, should be 'out/img/' folowed by a subdirectory
        named after the model name and parameters
    simple
        if set, don't show ticks, subtitle, labels
    save
        if True, save images to above path with filename t.jpg
    show
        if True, display the plot
    clear
        if True, clear the flot
    """
    plt.xlabel(t)
    if simple:
        plt.title(subtitle)
        plt.suptitle(title)

    if show:
        plt.show()
    if save:
        plt.savefig(f"{path}/{t}.jpg")
        plt.close()
    if clear:
        plt.cla()

    return

