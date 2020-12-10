#!/usr/bin/python
import numpy as np
from math import pi

from util.geometry import *

from typing import Any, Dict, List

class VicsekModel:
    """
    Setup the parameters for the Vicsek model simulation. The model simulates
    the behaviour of moving particles in a 2D continuous space. The only rule
    is: at each time step a given particle driven with a constant absolute
    velocity v assumes the average direction of motion of the particles in its
    neighborhood of radius r with some random perturbation E added.


    The position Xi of each particle i evolves as follows:

        Xi(t + dt) = Xi(t) + Vi(t) dt

    where the velocity Vi(t) is given by the absolute velocity v and angle Ai(t)
    which evolves as follows:

        Ai(t + i)  = <A(t)>(i, r) + dE

    For a detailed description of default parameter choices and initial conditions,
    Vicsek et al. (1995). "Novel Type of Phase Transition in a System of Self
    Driven Particles". Physical Review Letters. 75 (6): 1226–1229.
    https://arxiv.org/abs/cond-mat/0611743
    """

    def __init__(self,
                 n: int, l: int, e: float, bounded: bool,
                 v: float = 0.3, r: float = 1, dt: float = 1) -> None:
        """
        Initialise model with parameters, then create random 2D coordinate array
        X for the N particles, and random angle array A for the angle of their
        velocity

        Params
        ------
        n
            number of particles in the system
        l
            continuous space is LxL in size, with periodic boundaries
        e
            perturbation. Noise dE added in each evolution step is uniform
            distributed in [-E/2, E/2]
        v  = 0.3
            absolute velocity of each particle
        r  = 1
            proximity radius, normally used as distance unit
        dt = 1
            discrete time unit
        """
        self.n  = n
        self.l  = l
        self.e  = e
        self.v  = v
        self.r  = 1
        self.dt = 1

        self.X = np.random.uniform(0, l, size = (n, 2))
        self.A = np.random.uniform(-pi, pi, size = (n, 1))

        # we save the model name and params as a string, to be used when saving
        # and we also typeset a figure title
        rho = round(float(n) / l ** 2, 2)
        self.string = f"vicsek_eta{e}_rho{rho}"
        self.title  = f"$\eta$ = {e}, $\\rho$ = {rho}"

        # we count the time that has passed with every update
        self.t = 0


    def updateA(self, i: int) -> None:
        """
        Update angle of velocity for particle i by computing the average angle
        of all neighbouring particles and adding a perturbation dE

        Params
        ------
        i
            update angle Ai for particle with index i and location Xi at time t

        Side-effects
        ------
        update A[i] in the current object
        """
        indexes = neighbours(i, self.X, self.r, 'metric')
        Aavg    = np.average(self.A[indexes])

        dE = np.random.uniform(-self.e/2, self.e/2)

        self.A[i] = Aavg + dE


    def updateX(self, i: int) -> None:
        """
        Update coordinate for particle i adding new velocity to old coordinate

        If boundary reflection is enabled, and the particle would have crossed
        the boundary at the next timestep, then the particle is updated per the
        reflection rule instead of the generic rule.

        Params
        ------
        i
            update coordinate Xi for particle with index i and angle Ai at time t

        Side-effects
        ------
        update X[i]
        """

        self.updateA(i)
        Vi = ang_to_vec(self.A[i]) * self.v
        self.X[i] = bounds_wrap(self.X[i] + Vi * self.dt, self.l)


    def update(self) -> None:
        """
        Update every particle in the system to its new position and dump a file
        with the system state

        Side-effects
        ------
        update arrays A and X in current object
        increment counter t as a new timestep has passed
        """
        for i in range(self.n):
            self.updateX(i)

        self.t += self.dt

