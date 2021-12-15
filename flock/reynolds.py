#!/usr/bin/python
import numpy as np

from flock.model   import FlockModel
from util.geometry import *
from util.util     import save_var

from typing import Any, Dict, List, Tuple


class ReynoldsModel(FlockModel):
    """
    Setup the parameters for the Reynolds model simulation. The model simulates
    the behaviour of flocking birdoid objects (boids) in a 2D continuous space.
    In addition to the Vicsek model, the Reynolds model includes an aggegation
    and an avoidance parameter. The version used here is a simplified version of
    Reynold's [1] similar to the ones used by Seth [2] and Rosas et al [3], with
    boids behaving as particles with no mass, shape or dimension.

    The starting positions and velocities of boids are distributed uniformly at
    random.

    The position Xi of each boid i evolves as follows:

        Xi(t + dt) = Xi(t) + Vi(t) dt

    where the velocity Vi(t) is given by:

        Vi(t + dt) = Vi(t) + R1(i) + R2(i) + R3(i)

    where each of the three rules is regulated by scalar params a1, a2, a3.

    In the Reynolds model the neighbourhood is always metric and ignores certain
    angles aka (simulated perception). In this simplified version we use a
    convention from the Vicsek model, where metric neighbours are chosen with a
    parameter r that controls the range of the neighbourhood.

    [1] Reynolds CW (1987). "Flocks, Herds and Schools: A Distributed Behavioral
        Model". vol. 21. ACM. https://dl.acm.org/doi/10.1145/37402.37406
    [2] Seth AK (2010). "Measuring autonomy and emergence via Granger causality"
        Artificial Life. 16(2):179–196. pmid:20067405.
        https://doi.org/10.1162/artl.2010.16.2.16204
    [3] Rosas FE et al (2020). "Reconciling emergences: An information-theoretic
        approach to identify causal emergence in multivariate data". PLoS Comput
        Biol 16(12): e1008289. https://doi.org/10.1371/journal.pcbi.1008289
    """

    def __init__(self,
                 n: int, l: float,
                 bounds: EnumBounds, neighbours: EnumNeighbours,
                 a1: float, a2: float, a3: float, r: float = 1,
                 dt: float = 1
        ) -> None:
        """
        Initialise model with parameters, then create random 2D coordinate array
        X for the N boids, and random velocity vector array V for their velocity

        Params
        ------
        n
            number of boids in the system
        l
            continuous space is LxL in size, with periodic boundaries
        bounds
            enum value to specify whether boids wrap around boundaries
            (PERIODIC) or bounce off them (REFLECTIVE)
        neigbours
            enum value to whecify whether neighbourd are chosen if they are in a
            certain radius r from current boid (METRIC) or in the r closest
            neighbours (TOPOLOGICAl)
        a1
            avoidance: fly away from nearby boids
        a2
            alignment: align flight velocity to nearby boids
        a3
            aggregate: fly towards the center of mass of nearby boids
        r  = 1
            proximity radius, normally 1 if METRIC neighbours are used, or the
            number of neigbours to follow
        dt = 1
            discrete time unit
        """
        # initialise model-specific parameters
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.r  = r

        # initialise boid velocities with uniform random angles and speed
        angles = np.random.uniform(-np.pi, np.pi, size = (n, 1))
        self.V = np.array([ ang_to_vec(a) * np.random.uniform(0, 1)
                            for a in angles ])

        # initalise a generic flocking model and uniform positions of boids
        params = { 'alignment': a1, 'avoidance': a2, 'aggregate': a3, 'r': r }
        super().__init__('Reynolds', n, l, bounds, neighbours, dt, params)


    def __avoidance(self, i: int, indexes: List[int]) -> np.ndarray:
        """
        Ensure boids don't collide by subtracting the displacement of each nearby
        boid multiplied by the avoidance parameter.

        As per the original model by Reynolds, we assume avoidance is static
        (based on the relative position only).

        Params
        -----
        i
            index i for boid to update at time t
        indexes
            index list not including i for all of i's neighbours

        Returns
        ------
        np.ndarray of shape (D,)
        """
        disp = sum(relative_positions(self.X[indexes], self.X[i], self.l, self.bounds))

        return -disp * self.a1


    def __alignment(self, i: int, indexes: List[int]) -> np.ndarray:
        """
        Ensure boids match flight direction by adding the perceived velocity:
        from the velocity of i we subtract the average velocity of each nearby
        boid, and multiplby the alignment parameter.

        As per the original model by Reynolds, we assume alignment is dynamic (works
        on the velocity vector ignoring position).

        Params
        -----
        i
            index i for boid to update at time t
        indexes
            index list not including i for all of i's neighbours

        Returns
        ------
        np.ndarray of shape (D,)
        """
        vel = np.mean(self.X[indexes], axis = 0)

        return vel * self.a2


    def __aggregate(self, i: int, indexes: List[int]) -> np.ndarray:
        """
        Ensure boids fly towards the perceived centre of mass of nearby boids:
        from the centre of mass vector we subtract the position vector of current
        boid, and multiplby the aggregate parameter.

        Params
        -----
        i
            index i for boid to update at time t
        indexes
            index list not including i for all of i's neighbours

        Returns
        ------
        np.ndarray of shape (D,)
        """
        cmass = centre_of_mass(self.X[indexes], self.l, self.bounds)

        return cmass * self.a3



    def __new_X_V(self, i: int) -> Tuple[np.ndarray, float]:
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
        update velocity and position for particle i
        """
        indexes = neighbours(i, self.X, self.r, self.neighbours)
        # the current boid's velocity or posiiton is not considered
        indexes = [ j for j in indexes if j != i ]

        # update velocity only if there are any neighbours
        Vi = self.V[i]
        if indexes:
            v1 = self.__avoidance(i, indexes)
            v2 = self.__alignment(i, indexes)
            v3 = self.__aggregate(i, indexes)

            # find new velocity
            Vi += v1 + v2 + v3

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

        return (Xi, Vi)


    def update(self) -> None:
        """
        Update every boid in the system to its new position and velocity

        Side-effects
        ------
        update arrays V and X in current object
        increment counter t as a new timestep has passed
        """

        X_V = [ self.__new_X_V(i) for i in range(self.n) ]
        self.X = np.array([ X_V[i][0] for i in range(self.n) ])
        self.V = np.array([ X_V[i][1] for i in range(self.n) ])

        self.t += 1


    def save(self, path) -> None:
        """
        Save the state of every boid in the system (x-coord, y-coord of position
        and velocity) to the given path

        Side-effects
        ------
        if non-existent, create text files to save each variable
        append variable state to corresponding file
        """
        print(f'{self.t}: saving system state to {path}/')
        save_var(self.X[:,0], 'x1', path)
        save_var(self.X[:,1], 'x2', path)
        save_var(self.V[:,0], 'v1',  path)
        save_var(self.V[:,0], 'v2',  path)


