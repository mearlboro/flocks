#!/usr/bin/python
import numpy as np

from util.geometry import *

from typing import Any, Dict, List, Tuple

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
                 v: float = 0.1, r: float = 1, dt: float = 1) -> None:
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
        bounded
            if True, steer around area bounds, with radius r, else wrap-around
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
        self.r  = r
        self.dt = 1

        self.bounded = bounded

        self.X = np.random.uniform(0, l, size = (n, 2))
        self.A = np.random.uniform(-np.pi, np.pi, size = (n, 1))

        # we save the model name and params as a string, to be used when saving
        # and we also typeset a figure title
        rho = round(float(n) / l ** 2, 2)
        self.string = f"vicsek_eta{e}_rho{rho}"
        if bounded:
            self.string += '_bounded'
        self.title  = f"$\\eta$ = {e}, $\\rho$ = {rho}, $v$ = {v}, $r$ = {r}"

        # we count the time that has passed with every update
        self.t = 0

        print("Initialised " + "bounded " if bounded else " " + "Vicsek model")
        print("with parameters l: {l}, n: {n}, eta: {e}, v: {v}, r: {r}")


    def new_A(self, i: int) -> float:
        """
        Get updated angle of velocity for particle i by computing the average
        angle of all neighbouring particles (including itself) and adding a
        perturbation dE

        Params
        ------
        i
            index i for particle to update at time t

        Returns
        ------
        updated Ai
        """
        indexes = neighbours(i, self.X, self.r, 'metric')
        Aavg    = np.average(self.A[indexes])

        dE = np.random.uniform(-self.e/2, self.e/2)

        return Aavg + dE


    def new_X(self, i: int) -> Tuple[np.ndarray, float]:
        """
        Get updated coordinate and angle for particle i adding new velocity to
        old coordinate

        If boundary reflection is enabled, and the particle would have crossed
        the boundary at the next timestep, then the particle is updated per the
        reflection rule applied after the generic rule

        Params
        ------
        i
            index i for particle to update at time t

        Returns
        ------
        updated Xi, and updated Ai (as Ai might change when boundary reflection
        occurs)
        """

        # find new position and velocity according to normal rule
        Ai = self.new_A(i)
        Vi = ang_to_vec(self.A[i]) * self.v
        Xi = self.X[i] + Vi * self.dt

        # if it's out of bounds, correct based on simulation type
        if out_of_bounds(Xi, self.l):
            # for a toroidal world, new positions are wrapped around the space
            if not self.bounded:
                Xi = bounds_wrap(Xi, self.l)
            # otherwise, specular reflection happens against the walls
            else:
                (Xi, Vi) = bounds_reflect(Xi, Vi, self.dt, self.l)
                while out_of_bounds(Xi, self.l):
                    (Xi, Vi) = bounds_reflect(Xi, Vi, self.dt, self.l)

                Ai = vec_to_ang(Vi)

        return (Xi, Ai)


    def update(self) -> None:
        """
        Update every particle in the system to its new position and dump a file
        with the system state

        Side-effects
        ------
        update arrays A and X in current object
        increment counter t as a new timestep has passed
        """

        X_A = [ self.new_X(i) for i in range(self.n) ]
        self.X = np.array([  X_A[i][0]  for i in range(self.n) ])
        self.A = np.array([ [X_A[i][1]] for i in range(self.n) ])

        self.t += self.dt

