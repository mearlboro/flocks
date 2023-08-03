#!/usr/bin/python
import numpy as np

from util.geometry import *
from util.util     import save_var

from typing import Any, Dict, List, Tuple


class RandomWalker():
    """
    The strating positions of particles are distributed uniformly at random.

    The position Xi of each particle i evolves as follows after t timesteps of
    size dt, with noise for each term at each time E_i(t'):

        Xi(t) = Xi(0) + t*dt + sum_t' E_i(t')

    assuming E_i(t) ~ N(0, sigma^2)
    """

    def __init__(self, seed: int,
                 n: int, e: float,
                 dt: float = 1,
                 rand_state: bool = True,
                 start_state: np.ndarray = Null
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
        dt = 1
            time unit
        rand_state
            if set, start from random initial condiitons, otherwise start at 0
        """
        # initialise model-specific parameters
        self.n = n
        self.e = e
        self.dt = dt
        self.seed = seed

        # initialise seed
        np.random.seed(seed)

        # initialise positions
        #self.X0 = np.random.uniform(-1, 1, size = (n, 1))
        if start_state:
            self.X0 =  start_state
        elif rand_state:
            self.X0 = np.random.normal(0, 1, size = (n, 1))
        else:
            self.X0 = np.zeros(shape = (n, 1))
        self.X  = np.copy(self.X0)

        # trajectories
        self.t = 0
        self.traj = {}
        self.traj['X'] = []

        print(f"Initialised {n} 1D random walkers with seed {seed} and noise ~ N(0, {e**2})")


    def update(self) -> None:
        """
        Update every particle in the system to its new position after 1 timestep

        Side-effects
        ------
        update array X in current object
        increment counter t as a new timestep has passed
        """

        E = np.random.normal(0, self.e**2, size = (self.n, 1))
        X = self.X + self.dt + E

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
