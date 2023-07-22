#!/usr/bin/python

from pyflocks.models.model import FlockModel
from pyflocks.util.geometry import *
from pyflocks.util.util import save_var

from typing import Dict, List, Tuple


class ReynoldsModel(FlockModel):
    """
    Setup the parameters for the Reynolds model simulation. The model simulates
    the behaviour of flocking birdoid objects (boids) in a 2D continuous space.
    In addition to the Vicsek model, the Reynolds model includes an aggegation
    and an avoidance parameter. The version used here is a simplified version of
    Reynold's [1] similar to the ones used by Seth [2] and Rosas et al [3], with
    boids behaving as particles with no mass, shape or dimension. Each boid is
    characterised by its coordinates, angle of flight (in radians) and scalar
    speed.

    The position xi, yi of each boid i evolves as follows:

        xi(t + dt) = xi(t) + vi * cos(Ai(t)) * dt
        yi(t + dt) = yi(t) + vi * sin(Ai(t)) * dt

    where vi is the speed of particle i and the new angle of velocity Ai(t) is:

        Ai(t + dt) = Ai(t) + R1(i) + R2(i) + R3(i)

    where the three rules, using the scalar model params a1, a2, a3, regulate the
    behaviour of each boid with respect to its neighbours' proximity, direction,
    and centre of mass.
    cf. https://www.red3d.com/cwr/steer/gdc99/

    In the Reynolds model the neighbourhood is always metric and ignores certain
    angles at the back (a.k.a. simulated perception). In this simplified version
    we omit it and use the metric neighbourhood convention from the Vicsek model,
    where metric neighbours are chosen with parameter r that controls the range
    of the neighbourhood. The model can also be used with topological neigbours.

    [1] Reynolds CW (1987). "Flocks, Herds and Schools: A Distributed Behavioral
        Model". vol. 21. ACM. https://dl.acm.org/doi/10.1145/37402.37406
    [2] Seth AK (2010). "Measuring autonomy and emergence via Granger causality"
        Artificial Life. 16(2):179–196. pmid:20067405.
        https://doi.org/10.1162/artl.2010.16.2.16204
    [3] Rosas FE et al (2020). "Reconciling emergences: An information-theoretic
        approach to identify causal emergence in multivariate data". PLoS Comput
        Biol 16(12): e1008289. https://doi.org/10.1371/journal.pcbi.1008289
    """

    @property
    def default_params(self) -> Dict[str, float]:
        """
        Get model default parameters

            a1 (aggregate) = 0.15
                fly towards the center of mass of nearby boids
            a2 (avoidance) = 0.05
                fly away from nearby boids
            a3 (alignment) = 0.25
                align flight velocity to nearby boids
            r = 1
                proximity radius, normally 1 if METRIC neighbours are used, or the
                number of neigbours to follow
            minv = 3
                minimum starting absolute velocity
            maxv = 9
                maximum starting absolute velocity
        """
        return { 'aggregate': 0.15, 'avoidance': 0.05, 'alignment': 0.25,
                 'minv': 3, 'maxv': 9, 'r': 1 }


    def __init__(self, seed: int,
                 n: int, l: float,
                 bounds: EnumBounds, neighbours: EnumNeighbours,
                 dt: float = 0.1,
                 params: Dict[str, float] = {}
        ) -> None:
        """
        Initialise model with parameters, then create random 2D coordinate array
        X for the N boids. The starting positions, angles of velocity and speeds
        of boids are initialised with unformly random values as done by Seth.

        Params
        ------
        seed
            seed to be used for all random behaviour so that the simulation/
            experiment can be reproduced
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
        dt = 0.01
            time unit
        params
            dictionary of model-specific parameters, must contain:

            aggregate = 0.15
                fly towards the center of mass of nearby boids
            avoidance = 0.05
                fly away from nearby boids
            alignment = 0.25
                align flight velocity to nearby boids
            r = 1
                proximity radius, normally 1 if METRIC neighbours are used, or the
                number of neigbours to follow
            minv = 3
                minimum starting absolute velocity
            maxv = 9
                maximum starting absolute velocity
        """
        # merge with default params
        params = self.default_params | params

        # initalise a generic flocking model, seed, and uniform positions of boids
        super().__init__('Reynolds', seed, n, l, bounds, neighbours, dt, params)

        # initialise model-specific parameters
        self.r  = params['r']
        self.a1 = params['aggregate']
        self.a2 = params['avoidance']
        self.a3 = params['alignment']
        self.maxv = params['maxv']
        self.minv = params['minv']

        # initialise boid angle velocities with uniform random angles
        self.A = np.random.uniform(-np.pi, np.pi, size = n)

        # initialise boid absolute speeds with uniform random values in [minv,maxv]
        self.V = np.random.uniform(self.minv, self.maxv, size = n)



    def __aggregate(self, i: int) -> float:
        """
        Ensure boids fly towards the perceived centre of mass of ALL boids, i.e.
        the centre of mass excluding itself. To calculate the steering angle,
        the current position is subtracted from the perceived centre of mass
        before the difference angle is computed.
        """
        if self.n > 1:
            indexes = [ j for j in range(self.n) if j != i ]
            c = centre_of_mass(self.X[indexes], self.l, self.bounds)
            x = - relative_positions([ self.X[i]], c, self.l, self.bounds)[0]
            a = bearing_to(self.A[i], x)
            return a * self.a1
        else:
            return 0


    def __avoidance(self, i: int, indexes: List[int]) -> float:
        """
        Ensure boids don't collide by subtracting the displacement of nearby
        boids, normalised and weighted by the distance between each neighbour and
        the current boid.

        As per the original model by Reynolds, we assume avoidance is static
        (computation is based on the relative position only).
        """
        xs = - relative_positions(self.X[indexes], self.X[i], self.l, self.bounds)
        xs = [ x / metric_distance(x, self.X[i], self.l, self.bounds)**2
               for x in xs ]
        x  = np.sum(xs, axis = 0)
        a  = bearing_to(self.A[i], x)

        return a * self.a2


    def __alignment(self, i: int, indexes: List[int]) -> float:
        """
        Ensure boids match flight direction by adding the perceived velocity.
        from the mean velocity of neighbours we subtract the velocity of i and
        multiplby by the alignment parameter.

        As per the original model by Reynolds, we assume alignment is dynamic (works
        on the velocity vector ignoring position).
        """
        a = average_angles(self.A[indexes])
        a = ang_mod(a - self.A[i])

        return a * self.a3


    def __new_A(self, i: int) -> Tuple[np.ndarray, float, float]:
        """
        Get updated angle for particle i

        Params
        ------
        i
            index i for particle to update at time t

        Returns
        ------
        updated angle
        """
        indexes = neighbours(i, self.X, self.r, self.neighbours, self.bounds, self.l)
        # the current boid's velocity or position is not considered
        indexes = [ j for j in indexes if j != i ]

        # update velocity angle only if there are any neighbours
        Ai = self.A[i]

        Ai += self.__aggregate(i)
        if indexes:
            Ai += self.__avoidance(i, indexes)
            Ai += self.__alignment(i, indexes)

        return ang_mod(Ai)


    def __new_X_A_V(self, i: int) -> Tuple[np.ndarray, float, float]:
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
        update position, angle, speed for particle i
        """
        Ai = self.__new_A(i)

        # get velocity vector and normalise
        Vi = ang_to_vec(Ai) * self.V[i]
        v  = np.linalg.norm(Vi, 2)
        if v > self.maxv:
            Vi = Vi / v * self.maxv
            v  = self.maxv

        # get position at t+1
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

        return (Xi, Ai, v)


    def update(self) -> None:
        """
        Update every boid in the system to its new position and velocity

        Side-effects
        ------
        update arrays V and X in current object
        increment counter t as a new timestep has passed
        """

        X_A_V = [ self.__new_X_A_V(i) for i in range(self.n) ]
        self.X = np.array([ X_A_V[i][0] for i in range(self.n) ])
        self.A = np.array([ X_A_V[i][1] for i in range(self.n) ])
        self.V = np.array([ X_A_V[i][2] for i in range(self.n) ])

        self.t += self.dt


    def save(self, path) -> None:
        """
        Save the state of every boid in the system (x-coord, y-coord of position
        and velocity) to the given path

        Side-effects
        ------
        if non-existent, create text files to save each variable
        append variable state to corresponding file
        """
        super().save(path)

        save_var(self.X[:, 0], 'x1', path)
        save_var(self.X[:, 1], 'x2', path)
        save_var(self.A, 'a',  path)
        save_var(self.V, 'v',  path)


