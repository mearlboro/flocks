#!/usr/bin/python
import numpy as np

from flock.model   import FlockModel
from util.geometry import *
from util.util     import save_var

from typing import Any, Dict, List, Tuple


class VicsekModel(FlockModel):
    """
    Setup the parameters for the Vicsek model simulation. The model simulates
    the behaviour of moving particles in a 2D continuous space. The only rule
    is: at each time step a given particle driven with a constant absolute
    velocity v assumes the average direction of motion of the particles in its
    neighborhood of radius r with some random perturbation E added.

    The strating positions and velocities of particles are distributed uniformly
    at random.

    The position Xi of each particle i evolves as follows:

        Xi(t + dt) = Xi(t) + Vi(t) dt

    where the velocity Vi(t) is given by the absolute velocity v and angle Ai(t)
    which evolves as follows:

        Ai(t + i)  = <A(t)>(i, r) + dE

    For a detailed description of default parameter choices and initial conditions,
    [1] Vicsek et al. (1995). "Novel Type of Phase Transition in a System of Self
        Driven Particles". Physical Review Letters. 75 (6): 1226–1229.
        https://arxiv.org/abs/cond-mat/0611743
    """

    def __init__(self,
                 n: int, l: float,
                 bounds: EnumBounds, neighbours: EnumNeighbours,
                 e: float, v: float = 0.3, r: float = 1,
                 dt: float = 1
        ) -> None:
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
        v  = 0.3
            absolute velocity of each particle
        r  = 1
            proximity radius, normally 1 if METRIC neighbours are used, or the
            number of neighbours to follow
        dt = 1
            discrete time unit
        """
        # initialise model-specific parameters
        self.e = e
        self.v = v
        self.r = r

        # initialise particle velocity angles spread uniformly at random
        self.A = np.random.uniform(-np.pi, np.pi, size = (n, 1))

        # initalise a generic flocking model and uniform positions of particles
        params = { 'eta': e, 'v': v, 'r': r }
        super().__init__('Vicsek', n, l, bounds, neighbours, dt, params)


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
        indexes = neighbours(i, self.X, self.r, self.neighbours)
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

        self.t += 1


    def save(self, path) -> None:
        """
        Save the state of every particle in the system (x-coord, y-coord, angle)
        to the given path

        Side-effects
        ------
        if non-existent, create text files to save each variable
        append variable state to corresponding file
        """
        print(f'{self.t}: saving system state to {path}/')
        save_var(self.X[:,0], 'x1', path)
        save_var(self.X[:,1], 'x2', path)
        save_var(self.A[:,0], 'a',  path)


