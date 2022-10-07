#!/usr/bin/python
import numpy as np

from flock.model   import FlockModel
from util.geometry import *
from util.util     import save_var

from typing import Any, Dict, List, Tuple


class KuramotoVicsekModel(FlockModel):
    """
    This simulation combines a Vicsek 2D particle model with a Kuramoto model of
    oscillators.

    At each time step a given particle driven with a constant absolute velocity
    v assumes the average direction of motion of the particles in its
    neighborhood of radius r with some random perturbation E added.

    The position Xi of each particle i evolves as follows:

        Xi(t + dt) = Xi(t) + Vi(t) dt

    where the velocity Vi(t) is given by the absolute velocity v and angle Ai(t)
    which evolves into the average angle <A(t)> of its neigbours:

        Ai(t + dt)  = <A(t)> + dE

    For a description of the self-propelled particle model see Vicsek et al. (1995).
    "Novel Type of Phase Transition in a System of Self Driven Particles".
    Physical Review Letters. 75 (6): 1226–1229.
    https://arxiv.org/abs/cond-mat/0611743

    Moreover, each particle acts as an oscillator with frequency Fi, phase Pi.
    These oscillators are coupled with coupling parameter k if they are in
    proximity of each other, and the phase Pi of each oscillator evolves as:

        Pi(t + dt) = Pi + dP * dt

    where the phase difference dP comes from the angular frequency and the sum
    of angular differences with its neighbours weighted by the parameter k:

        dP = Fi * 2pi + k * sum(sin(Pi(t) - Pj(t))

    It is assumed that each oscillator has the same frequency f, but this can
    be easily amended through the use of the array F.

    For a description of the oscillator model see,  Kuramoto, Y. (1984).
    "Chemical Oscillations, Waves, and Turbulence."
    https://doi.org/10.1007/978-3-642-69689-3
    """

    def __init__(self, seed: int,
                 n: int, l: int,
                 bounds: EnumBounds, neighbours: EnumNeighbours,
                 e: float, v: float = 0.3, r: float = 1,
                 k: float = 1, f: float = 1,
                 dt: float = 0.1
        ) -> None:
        """
        Initialise model with parameters, then create random 2D coordinate array
        X for the N particles, and random angle arrays A and P for the angles of
        their velocity and the starting phase of the oscillator

        Params
        ------
        seed
            seed to be used for all random behaviour so that the simulation/
            experiment can be reproduced
        n
            number of particles in the system
        l
            continuous space is LxL in size, with boundaries specified by the
            `boundaries` param
        bounds
            enum value to specify whether particles wrap around boundaries
            (PERIODIC) or bounce off them (REFLECTIVE)
        neigbours
            enum value to whecify whether neighbourd are chosen if they are in a
            certain radius r from current particle (METRIC) or in the r closest
            neighbours (TOPOLOGICAl)
        e
            perturbation. Noise dE added in each evolution step is uniform
            distributed in [-E/2, E/2]
        v  = 1
            absolute velocity of each particle
        r  = 1
            proximity radius, normally used as distance unit, or number of
            neighbours to follow
        k
            for the Kuramoto model, the coupling parameter between neighbours
        f  = 1
            frequency of oscillators, measured in Hertz
        dt
            time unit
        """
        # initialise model-specific parameters
        self.e  = e
        self.v  = v
        self.r  = r
        self.f  = f
        self.k  = k

        # initialise seed
        np.random.seed(seed)

        # initialise particle velocity angles spread uniformly at random
        self.A = np.random.uniform(-np.pi, np.pi, size = (n, 1))

        # initialise oscillator frequency and phase
        self.F = np.zeros(shape = (n)) + f
        self.P = np.random.uniform(0, np.pi, size = (n))

        # initalise a generic flocking model and uniform positions of particles
        params = { 'eta': e, 'v': v, 'r': r, 'k': k, 'f': f }
        super().__init__('KuramotoVicsek', seed, n, l, bounds, neighbours, dt, params)


    def __new_A(self, i: int) -> float:
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
        indexes = neighbours(i, self.X, self.r, self.neighbours, self.bounds, self.l)
        Aavg = average_angles(self.A[indexes])
        dE = np.random.uniform(-self.e/2, self.e/2)

        return Aavg + dE


    def __new_X(self, i: int) -> Tuple[np.ndarray, float]:
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
        Ai = self.__new_A(i)
        Vi = ang_to_vec(Ai) * self.v
        Xi = self.X[i] + Vi * self.dt

        # if it's out of bounds, correct based on simulation type
        if out_of_bounds(Xi, self.l):
            # for a toroidal world, new positions are wrapped around the space
            if self.bounds == EnumBounds.PERIODIC:
                Xi = bounds_wrap(Xi, self.l)
            # otherwise, specular reflection happens against the walls
            else:
                (Xi, Vi) = bounds_reflect(Xi, Vi, self.l)
                while out_of_bounds(Xi, self.l):
                    (Xi, Vi) = bounds_reflect(Xi, Vi, self.l)

                Ai = vec_to_ang(Vi)

        return (Xi, Ai)


    def __new_P(self, i: int) -> float:
        """
        Compute new phase P for particle i based on Kuramoto formula, when only
        coupled to the nearest neighbours

        Params
        ------
        i
            index i for particle to update at time t

        Returns
        ------
        updated Pi, a float between 0 and pi
        """
        indexes = neighbours(i, self.X, self.r, self.neighbours, self.bounds, self.l)

        Pi = self.P[i]
        Wi = 2 * np.pi * self.F[i]

        dP = Wi + self.k * sum(
            [ np.sin(self.P[j] - self.P[i]) for j in indexes ])
        Pi = Pi + dP * self.dt

        return ang_mod(Pi)



    def update(self) -> None:
        """
        Update every particle in the system to its new position

        Side-effects
        ------
        update arrays A and X in current object
        increment counter t as a new timestep has passed
        """

        X_A = [ self.__new_X(i) for i in range(self.n) ]
        self.X = np.array([  X_A[i][0]  for i in range(self.n) ])
        self.A = np.array([ [X_A[i][1]] for i in range(self.n) ])
        self.P = np.array([ self.__new_P(i) for i in range(self.n) ])

        self.t += self.dt


    def save(self, path) -> None:
        """
        Save the state of every particle in the system (x-coord, y-coord, angle)
        to the given path

        Side-effects
        ------
        if non-existent, create text files to save each variable
        append variable state to corresponding file
        """
        super().save(path)

        save_var(self.X[:,0], 'x1', path)
        save_var(self.X[:,1], 'x2', path)
        save_var(self.A[:,0], 'a',  path)
        save_var(self.P, 'p',  path)

