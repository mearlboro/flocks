#!/usr/bin/python
import numpy as np
import re
import os

from util.geometry import EnumBounds, EnumNeighbours
from util.util     import load_var

from typing import Any, Dict, List, Tuple


class FlockModel:
    """
    Setup the parameters for a flocking model simulation. The model simulates
    the behaviour of moving particles in a 2D continuous space in discrete time
    steps.

    The particles are initialised in the space with model-specific positions and
    velocity vectors. At each discrete time step, the position (and velocity)
    may be updated according to model rules.
    """
    def __init__(self, name: str, seed: int, n: int, l: float,
                 bounds: EnumBounds, neighbours: EnumNeighbours,
                 dt: float = 1,
                 params: Dict[str, float] = {}) -> None:
        """
        Initialise model with parameters, then create 2D coordinate array X for
        the N particles in the lxl space. The coordinates are distributed
        uniformly at random unless the model specifies otherwise.

        Params
        ------
        name
            name of the model
        seed
            seed to be used for all random behaviour so that the simulation/
            experiment can be reproduced
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
        dt = 1
            discrete time unit
        params
            dictionary containing parameter names and values for the model
        """
        self.n  = n
        self.l  = l
        self.dt = dt

        self.bounds = bounds
        bounds_str  = bounds.name.lower()
        self.neighbours = neighbours
        neighbours_str  = neighbours.name.lower()

        # initialise parameters
        rho = round(float(n) / l ** 2, 4) # density
        params['rho'] = rho

        self.params = params
        params_strs = [ f'{p}{v}' for p,v in params.items() ]

        # we save the model name and params as a string, to be used when saving
        # and we also typeset a figure title and subtitle
        self.string   = f"{name}_{bounds_str}_{neighbours_str}_{'_'.join(params_strs)}_{seed}"
        self.title    = f"{name} model, {bounds_str} bounds, {neighbours_str} neighbours"
        self.subtitle = ', '.join([ f'${p}$ = {v}' if len(p) not in range(3, 7)
                                    else f'$\\{p}$ = {v}'
                                    for p,v in params.items() ])

        # initialise seed
        np.random.seed(seed)

        # initialise particle positions spread uniformly at random
        self.X = np.random.uniform(0, l, size = (n, 2))

        # initialise time and trajectories
        self.t    = 0
        self.traj = {}

        print(f"Initialised {self.title}, n = {self.n}, l = {self.l}, dt = {self.dt}")
        print(f" with parameters: {self.subtitle}")
        print(f" and seed {seed}")


    @classmethod
    def load(cls, path: str) -> 'FlockModel':
        """
        Factory method to initialise a simulated model using information at the
        given path. The folder contains trajectories for all relevant variables
        in the model and is named according to model params.

        Variables are stored as numpy 2D array of shape (T, N) or (T, N, D), where
        X[t, i] or X[t, i, :] is the value of system variable Xi at time t

        This function will initialise the model object as well as append each
        state at each timestep to a 4-dimensional trajectory numpy array

        Params
        ------
        path
            system path to a folder containing the state of model variables, as
            returned by `mkdir`, to be parsed to extract model parameters

                {name}_{bounds}_{neighbours}(_{paramname}{paramvalue})+(-{simID})?

            for example, if the root output path is '/out/txt', a Vicsek model
            with 10 particles in a 1x1 space, with periodic boundaries, metric
            neighbours and params rho = 0.1, eta = 0.5, r = 1 will use the path

                out/txt/Vicsek_periodic_metric_rho0.1_eta0.5_r1

        Returns
        ------
        a FlockModel with all params initialised and trajectories
        """

        if not os.path.isdir(path):
            raise ValueError('No folder at the given path')
            exit(0)

        if path[-1] == '/':
            path = path[:-1]

        # parse the directory name to extract model parameters, excluding the ID
        d  = os.path.basename(path).split('-')[0]
        ps = d.split('_')
        ps_dict = { re.findall('[a-z]+', p)[0]: float(re.findall('[0-9.]+', p)[0])
                    for p in ps[3:] }

        print(f'Loading {ps[0]} model from {path} with params {ps_dict}')
        # loads all .txt files in the folder
        files = [f for f in os.listdir(path)
                   if os.path.isfile(os.path.join(path, f))
                   and f.split('.')[-1] == 'txt' ]
        var_dict = {}
        if files:
            var_dict = { f.split('.')[0].upper(): load_var(os.path.join(path, f))
                        for f in files }

        # variables named x1, x2...  denote coordinates and should be combined
        Xvars = [ var_dict[x] for x in sorted(var_dict.keys()) if 'X' in x ]

        # get number of agents, timesteps, and system size from the array shapes
        (t, n) = Xvars[0].shape
        l = np.sqrt(n / ps_dict['rho'])

        # call constructor with the params above
        model = cls(ps[0], n, l, EnumBounds[ps[1].upper()],
             EnumNeighbours[ps[2].upper()], 1, ps_dict)

        # then store variable trajectories in a trajectory dictionary
        model.traj['X'] = np.array([ np.stack([ X[i] for X in Xvars ], axis = 1)
                                  for i in range(t) ])
        for var in var_dict.keys():
            if 'X' not in var:
                model.traj[var] = var_dict[var]

        return model



    @property
    def trajectories(self) -> Dict[str, np.ndarray]:
        """
        Return trajectories for a system simulation loaded from file
        """
        return self.traj

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Return a hash of all model parameters
        """
        return self.params


    def update(self) -> None:
        """
        Update every particle in the system to its new position
        """
        pass


    def save(self, path) -> None:
        """
        Save state of every particle in the system to file
        """
        print(f'{int(self.t / self.dt)}: saving system state to {path}')


    def mkdir(self, root_dir) -> str:
        """
        Create output folder based on simulation name to store simulation
        results with a name of the form

            {root_dir}/
                {name}_{bounds}_{neighbours}(_{paramname}{paramvalue})+(-{simID})?_{seed}

        """
        pth = f'{root_dir}/{self.string}'

        while os.path.isdir(pth):
            # get the last sim ID
            if '-' in pth:
                [prefix, str_id] = pth.split('-')
                sim_id = int(str_id) + 1
                pth = f'{prefix}-{sim_id}'
            else:
                pth = f'{pth}-1'

        os.mkdir(pth)

        return pth

