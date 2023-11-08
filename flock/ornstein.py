#!/usr/bin/python
import numpy as np

from util.geometry import *
from util.util     import save_var

from typing import Any, Dict, List, Tuple


class OUWalker():
    """
    This is a version of the Gaussian random walker but which is designed to be
    stationary.

    The starting positions of particles are always 0 unless stated otherwise.

    The position Xi of each particle i evolves as follows after t timesteps of
    size dt, with noise for each term at each time Ei(t). Moreover the particle
    is 'dragged back' to origin with constant k.

        Xi(t+1) = Xi(t) + Ei(t) * dt - k X_i(t)


    If there is any coupling, then each particle is updated according to the
    positions of its neighbours and the coupling g:

        Xi(t+1) = Xi(t) + g(X[i-1](t) + X[i+1](t) - 2 Xi(t)) + Ei(t)

    for i in range(2, N-1).

    Open boundaries are given by

        x[1](t+1) = x[1](t) + g(x[2](t)   - x[1](t) - a) + E1(t)
        x[N](t+1) = x[N](t) + g(x[N-1](t) - x[N](t) + a) + EN(t)

    """
    def __init__(self, seed: int,
                 n: int, e: float, k: float,
                 g: float = 0, a: float = 1.0, dt: float = 1.0, dx: float = 0.0,
                 rand_state: bool = True,
                 start_state: np.ndarray = None
        ) -> None:
        """
        Initialise model with parameters, then create random 1D coordinate array
        X for the N particles, and random 1D array for the noise samples.
        Space is 1D infinite.

        Params
        ------
        seed
            seed to be used for all random behaviour so that the simulation/
            experiment can be reproduced
        n
            number of particles in the system
        e
            perturbation. Noise E added in each evolution step is Gaussian
            distributed in N(0, e^2)
        k
            restoring force strength: pulls back the walkers towards the
            starting position
        g
            coupling between a particle and its neighbours.
        dt = 1
            time unit
        dx = 0
            distance moved in one time unit
        a = 1
            initial distance between particles, if rand_state and start_state
            not set
        rand_state
            if set, start from random initial condiitons, otherwise start at
        start_state
            and otherwise all particles start at i*a
        """
        # initialise model-specific parameters
        self.n = n
        self.e = e
        self.g = g
        self.a = a
        self.k = k
        self.dt = dt
        self.dx = dx
        self.seed = seed

        # initialise seed
        np.random.seed(seed)

        # initialise positions
        if start_state is not None:
            self.X0 =  start_state
        elif rand_state:
            self.X0 = np.random.normal(0, 1, size = (n, 1))
        else:
            self.X0 = np.arange(1, n + 1) * a
            self.X0 = self.X0.reshape((n, 1))
        self.X  = np.copy(self.X0)

        # trajectories
        self.t = 0
        self.traj = {}
        self.traj['X'] = []

        print(f"Initialised {n} 1D random walkers with seed {seed}, coupling {g}, restoring force {k} and noise ~ N(0, {round(e**2, 4)})")


    def update(self) -> None:
        """
        Update every particle in the system to its new position after 1 timestep

        Side-effects
        ------
        update array X in current object
        increment counter t as a new timestep has passed
        """
        n = self.n
        X = np.copy(self.X)
        # compute noise
        E = np.random.normal(0, self.e**2, size = (n, 1))
        # compute the restoring force
        K = (X - self.X0) * self.k
        # compute couplings
        C = np.zeros((n, 1))
        C[1:n-1] = X[0:n-2] + X[2:n] - 2 * X[1:n-1]
        C[0]     = X[1]     - X[0]   - self.a
        C[n-1]   = X[n-2]   - X[n-1] + self.a
        C *= self.g

        # update positions
        X += self.dt * (self.dx - K + C + E)

        self.X = np.copy(X)
        self.t += self.dt
        self.traj['X'].append(X)


    def save(self, path) -> None:
        """
        Save the state of every particle in the system (x-coord, y-coord, angle)
        to the given path

        Side-effects
        ------
        if non-existent, create text files to save each variable
        append variable state to corresponding file
        """
        save_var(self.X[:], 'x', path)


if __name__ == "__main__":
    rw = RandomWalker(0, 10, 1)
    times = range(1000)

    for _ in times:
        rw.update()
    traj = np.array(rw.traj)
    cmass = np.mean(traj, axis = 1)

    import matplotlib.pyplot as plt
    for i in range(10):
        plt.plot(times, traj[:, i], label = f"$x_{i}(t)$", linewidth = 1, alpha = 0.8)
    plt.plot(times, cmass, label = f"R(t)", linewidth = 3, color='grey')
    plt.ylabel("Position of $x_i$")
    plt.xlabel("time")
    plt.legend()
    plt.show()
