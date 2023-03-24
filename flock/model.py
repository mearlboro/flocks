#!/usr/bin/python
from abc import abstractmethod
import numpy as np
import re
import os

from util.geometry import EnumBounds, EnumNeighbours
from util.util     import load_var

from typing import Any, Dict, List, Tuple


class Flock:
    """
    An object to hold the results of an experimental trial which has similar
    data to the simulations in FlockModel.

    Assuming that experimental data is given in the same format as the simulation
    outputs, go through given directory and load trajectories and angles from an
    experiment in the same data structures as trajectories and angles from a
    simulation.

    Experimental data assumes l = 1, and interactions are unknown, so there is
    no equivalent for r and neighbourhood is unknown. The time unit is also
    assumed to be discrete dt = 1 unless stated otherwise. Bounds are always
    reflective.
    """
    def __init__(self, name: str, segment: str, n: int, l: float,
            dt: float = 1, params: dict = {}
        ) -> None:
        """
        Initialise and populate class with known parameters of an experiment and
        the trajectories, angles and velocities of flocking behaviour obtained
        during the experiment.

        Params
        ------
        name
            name of the experiment
        segment
            name of segment or trial to differentiate between different instances
            of the experiment
        n
            number of particles in the system
        l
            continuous space is LxL in size, with reflective boundaries
        dt
            time unit
        params
            dictionary containing any useful parameters for the experiment that
            should be tracked down during analysis
        """
        self.n  = n
        self.l  = l
        self.dt = dt
        self.r  = 0

        self.bounds = EnumBounds.REFLECTIVE
        self.neighbours = EnumNeighbours.UNKNOWN

        # initialise parameters
        rho = round(float(n) / l ** 2, 4) # density
        params['rho'] = rho

        self.params = params

        # dir name, figure title and subtitle
        self.string   = f"{name}_{segment}"
        self.title    = f"{name} experiment {segment}: {n} subjects"
        self.subtitle = ', '.join([ f'{p} = {v}' if len(p) != 3
                                    else f'$\\{p}$ = {v}'
                                    for p,v in params.items() ])
        self.t    = 0
        self.traj = {}


    @classmethod
    def load(cls, path: str, dt: float = 1, params: dict = {}) -> 'Flock':
        """
        Static class method to create a flock-like object with trajectories produced
        by an experment using information at the given path. The folder contains
        trajectories for all relevant variables and is named according to the
        experiment's name and segment/trial from which data was collected.

        Variables are stored as numpy 2D array of shape (T, N) or (T, N, D), where
        X[t, i] or X[t, i, :] is the value of system variable Xi at time t

        This function will initialise the object as well as append each
        state at each timestep to a 4-dimensional trajectory numpy array

        It is likely that the experimental data available may be just of positions
        but angles and velocities are also relevant so they are to be constructed
        based on the positions.

        Params
        ------
        path
            system path to a folder containing experimental data

        Returns
        ------
        a Flock with trajectories
        """

        if not os.path.isdir(path):
            raise ValueError('No folder at the given path')
            exit(0)

        if path[-1] == '/':
            path = path[:-1]

        name, seg = path.split('/')[-1].split('_')[:2]

        X1t = load_var(f"{path}/x1.txt")
        X2t = load_var(f"{path}/x2.txt")
        At  = load_var(f"{path}/a.txt")
        Vt  = load_var(f"{path}/v.txt")
        (t, n) = At.shape

        flock = Flock(name, seg, n, 1, dt)
        flock.t = t
        flock.traj['X'] = np.array([ np.stack([ X[i]
                            for X in [X1t, X2t] ], axis = 1)
                            for i in range(t) ]).astype(float)
        flock.traj['A'] = At
        flock.traj['V'] = Vt

        return flock


    def mkdir(self, root_dir) -> str:
        """
        Create output folder based on experiment name to store trajectories in
        plain text with a name of the form

            {root_dir}/
                {name}(_{info})+(_{paramname}{paramval})+_{seed}?-id

        The IDs are used in particular for images in out/plt as there may be
        multiple visualisations for the same simulation. Otherwise, multiple
        simulations with the same parameters are to be differentiated by seed.
        """
        pth = f'{root_dir}/{self.string}'

        if 'img' in root_dir:
            while os.path.isdir(pth):
                # get the last sim ID
                if '-' in pth:
                    [prefix, str_id] = pth.split('-')
                    sim_id = int(str_id) + 1
                    pth = f'{prefix}-{sim_id}'
                else:
                    pth = f'{pth}-1'
            os.mkdir(pth)
        else:
            if not os.path.isdir(pth):
                os.mkdir(pth)
        return pth




class FlockModel(Flock):
    """
    Setup the parameters for a flocking model simulation. The model simulates
    the behaviour of moving particles in a 2D continuous space in discrete time
    steps.

    This is the generic constructor, which gets called by each child class (type
    of flocking model) when initialising a model object and the simulation.

    The particles are initialised in the space with model-specific positions and
    velocity vectors. At each discrete time step, the position (and velocity)
    may be updated according to model rules.
    """
    def __init__(self, name: str, seed: int, n: int, l: float,
                 bounds: EnumBounds, neighbours: EnumNeighbours,
                 dt: float = 1,
                 params: Dict[str, float] = {}
        ) -> None:
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
            continuous space is LxL in size
        bounds
            enum value to specify whether particles wrap around boundaries
            (PERIODIC) or bounce off them (REFLECTIVE)
        neigbours
            enum value to whecify whether neighbourd are chosen if they are in a
            certain radius r from current particle (METRIC) or in the r closest
            neighbours (TOPOLOGICAl)
        dt
            time unit
        params
            dictionary containing parameter names and values for the model
        """
        super().__init__('', '', n, l, dt, params)

        self.name = name.capitalize()
        self.bounds = bounds
        bounds_str  = bounds.name.lower()
        self.neighbours = neighbours
        neighbours_str  = neighbours.name.lower()

        # we save the model name and params as a string, to be used when saving
        # and we also typeset a figure title and subtitle
        params_strs = [ f'{p}{v}' for p,v in self.params.items() ]
        self.string   = f"{self.name}_{bounds_str}_{neighbours_str}_{'_'.join(params_strs)}_{seed}"
        self.title    = f"{self.name} model, {bounds_str}, {neighbours_str}"

        # initialise seed
        if seed >= 0:
            np.random.seed(seed)

        # initialise particle positions spread uniformly at random
        self.X = np.random.uniform(0, l, size = (n, 2))

        print(f"Initialised {self.title}, n = {self.n}, l = {self.l}, dt = {self.dt}")
        print(f"  with parameters: {self.subtitle}")
        print(f"  and seed {seed}")


    @classmethod
    def __load(cls, path: str) -> 'FlockModel':
        """
        Initialise a simulated model using information at the given path. The
        folder contains trajectories for all relevant variables in the model and
        is named according to model params.

        Variables are stored as numpy 2D array of shape (T, N) or (T, N, D), where
        X[t, i] or X[t, i, :] is the value of system variable Xi at time t

        This function will initialise the model object as well as append each
        state at each timestep to a 4-dimensional trajectory numpy array.

        This function is called by the child classes of each model type.

        Params
        ------
        path
            system path to a folder containing the state of model variables, as
            returned by `mkdir`, to be parsed to extract model parameters

                {name}_{bounds}_{neighbours}(_{paramname}{paramvalue})+_seed(-{simID})?

            for example, if the root output path is '/out/txt', a Vicsek model
            with 10 particles in a 1x1 space, with periodic boundaries, metric
            neighbours and params rho = 0.1, eta = 0.5, r = 1 and seed 999 will
            use the path

                out/txt/Vicsek_periodic_metric_rho0.1_eta0.5_r1_999

        Returns
        ------
        a FlockModel with all params initialised and trajectories
        """

        if not os.path.isdir(path):
            raise ValueError('No folder at the given path')
            exit(0)

        if path[-1] == '/':
            path = path[:-1]

        # parse directory name to extract model parameters, exclude seed and ID
        d  = os.path.basename(path).split('-')[0]
        ps = d.split('_')
        seed = ps[-1]

        # some simulations may not have seed information
        try:
            seed = int(seed)
            ps = ps[:-1]
        except:
            seed = -1

        ps_dict = { re.findall('[a-z]+', p)[0]: float(re.findall('[0-9.]+', p)[0])
                    for p in ps[3:]
                    if len(re.findall('[0-9.]+', p)) }

        print(f"Found {ps[0]} model with params {ps_dict} and seed {seed}")
        # loads all .txt files in the folder
        files = [f for f in os.listdir(path)
                   if os.path.isfile(os.path.join(path, f))
                   and f.split('.')[-1] == 'txt' ]
        var_dict = {}
        if files:
            var_dict = { f.split('.')[0].upper(): load_var(os.path.join(path, f))
                        for f in files }
        else:
            print(f"No .txt files storing simulation results were found at {path}")

        # variables named x1, x2...  denote coordinates and should be combined
        Xvars = [ var_dict[x] for x in sorted(var_dict.keys()) if 'X' in x ]

        # get number of agents, timesteps, and system size from the array shapes
        (t, n) = Xvars[0].shape
        l = np.sqrt(n / ps_dict['rho'])

        print(f"Found time series of length t={t} for n={n} variables")

        model = cls(seed, n, l, EnumBounds[ps[1].upper()],
             EnumNeighbours[ps[2].upper()], 1, ps_dict)

        # then store variable trajectories in a trajectory dictionary
        model.traj['X'] = np.array([ np.stack([ X[i] for X in Xvars ], axis = 1)
                                  for i in range(t) ])
        for var in var_dict.keys():
            if 'X' not in var:
                model.traj[var] = var_dict[var]

        model.t = t

        return model


    @classmethod
    def load(self, path: str) -> 'FlockModel':
        """
        Load simulation data from path where trajectories are stored in .txt
        """
        if type(self) == FlockModel:
            raise NotImplementedError("Cannot initialise FlockModel directly. Use specific model constructor or the FlockFactory object")
        else:
            return self.__load(path)


    @abstractmethod
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


