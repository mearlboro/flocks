# Library for simulating flocks

Implemented in Python and packaged with `pipenv`. Use

```sh
$ pipenv install -r requirements.txt
$ pipenv shell
$ python -m flock.main {model_name} --help
```

for information on simulation parameters, and run also with `python -m`. Default
parameters are specified in `main.py`.

Simulation results are stored in `out/txt`.

The utility `util/animate.py` takes a simulation output in `out/txt` and plots
the state of the simulation at each timestep and saves it in `out/img`.
The plotting libraries are located in `util/plot.py`.

Use `./gif.sh` to turn all the images into animations of the model. Requires
`ffmpeg` to be installed.

## Model Simulations & Animations
Currently the Standard Vicsek Model (SVM) is implemented reliably, plus topological
interactions and reflective boundaries.

A variant of the model is also implemented where each particle is a Kuramoto
oscillator. This has not been fully tested to full scrutiny.

Reynolds flocking model is also in progress, but it does NOT behave correctly.

Each simulation has a random seed saved in the directory name aside with the
parameters, so it is fully reproducible.

## Examples
```sh
$ python -m flock.main vicsek -e 0.4 -n 100 -l 5 --bounds PERIODIC --neighbours TOPOLOGICAL
$ python -m util.animate out/txt/Vicsek_periodic_topological_eta0.4_v0.1_r1.0_rho4.0_{seed} \
     --style ARROW
$ cd out
$ ./gif.sh
```
![](/out/gif/Vicsek_periodic_topological_eta0.4_v0.1_r1.0_rho4.0.gif)

```sh
$ python -m flock.main vicsek -e 1 -n 20 -l 2 --bounds PERIODIC --neighbours METRIC
$ python -m util.animate out/txt/Vicsek_periodic_metric_eta1.0_v0.1_r1.0_rho5.0_{seed} \
     --style DOT --traj 20 --cmass
$ cd out
$ ./gif.sh
```
![](/out/gif/Vicsek_periodic_metric_eta1.0_v0.1_r1.0_rho5.0_traj.gif)

```sh
$ python -m flock.main vicsek -e 0.7 -n 20 -l 2 --bounds REFLECTIVE --neighbours TOPOLOGICAL
$ python -m util.animate out/txt/Vicsek_reflective_topological_eta0.7_v0.1_r3.0_rho5.0_{seed} \
     --style ARROW --sumvec
$ cd out
$ ./gif.sh
```
![](/out/gif/Vicsek_reflective_topological_eta0.7_v0.1_r3.0_rho5.0_sumvec.gif)

# Analysis & Plots

Code located in `analysis/` is for computing statistics, order parameters and
emergence measures on the exported simulation trajectories in `out/txt`.

Moreover, generic trajectory data in the same format that may have not been
generated by the library can be loaded using functions in `flock/factory.py`.

There is support for generating analysis plots in `analysis/plot.py`.

## Order parameters

`anaysis/order.py` contains simple implementations of instantaneous order parameters
applicable to an array of system states. These can be from the same simulation
or multiple realisations in an ensemble. Results are stored in `out/order`.

### Analysing one simulation
Running
```sh
$ python -m analysis.order --path out/txt/$simulation_folder --ordp $ordp_name
```
will compute the order parameter specified for a given simulation.

### Analysing multiple simulated realisations of the same model
The functions in `order.py` are used also in `ensemble.py` to automate computing
order parameters and obtain ensemble averages while keeping track of control
parameters. Run as

```sh
$ python -m analysis.ensemble --path out/txt --name {model_name} --ordp {ordp_name} \
    -p {param1} -p {param2} [-p {param3}]
```
which will save data with the model and param names in `out/order` and also produce
a plot in `out/img`.

## Examples

To produce the plot below, after 10 simulations were created with the same `n`,
`rho`, and a range of `eta`, where the error is the standard error for the order
parameter for the models:

```sh
$ python -m analysis.ensemble --path out/txt/ --ordp VICSEK_ORDER \
    --name 'Vicsek_reflective_metric' -p rho -p eta
```

![](/out/plt/Vicsek_reflective_metric_rho4.0_eta_vs_vicsek_order.png)

In case 3 aggregate parameters are passed, one plot will be made for each value of the first, plotting statistics against values of the second, for each value of the third e.g

```sh
$ python -m analysis.ensemble --path out/txt/ --ordp VICSEK_ORDER \
    --name 'Vicsek_reflective_topological' -p rho -p eta -p r
```

![](/out/plt/Vicsek_reflective_topological_rho2.5_eta_r_vs_avg_abs_vel.png)

## Computing flock emergence

To test if an order parameter or supervenient feature of the system is an
emergent feature of the system, we use the synergy-redundancy index between
parts and whole, based on the theory of causal emergence of Rosas et al (2020).
`analysis/emergence.py` contains functions which use the `infodynamics.jar`
binary of the Java Information Dynamics Toolkit (JIDT) to estimate mutual
informations between time series. The aim is to apply the causal emergence
theory to trajectories or angles and some order parameter.

To run an example emergence computation on some testing Gaussian data and the sum of the two time
series with the Gaussian estimator:

```sh
$ python -m analysis.emergence --est Gaussian
```

or on an existing model's trajectories with the centre of mass, say `Vicsek_periodic_metric_eta0.1_v0.1_r1.0_rho1.0`

```sh
$ python -m analysis.emergence --model out/img/Vicsek_periodic_metric_eta0.1_v0.1_r1.0_rho1.0
    --est Kraskov1
```

# References

Vicsek T, Czirók A, Ben-Jacob E, Cohen I, Shochet O (1995). _"Novel Type of Phase Transition in a System of Self-Driven Particles"_. Physical Review Letters. 75 (6): 1226–1229. arXiv:cond-mat/0611743.
doi:10.1103/PhysRevLett.75.1226.

Kuramoto Y (1984). _"Chemical Oscillations, Waves, and Turbulence"_. New York, NY: Springer-Verlag.

Reynolds C (1987). _"Flocks, herds and schools: A distributed behavioral model"_.
SIGGRAPH '87: Proceedings of the 14th Annual Conference on Computer Graphics and Interactive Techniques.
Association for Computing Machinery. pp. 25–34. doi:10.1145/37401.37406.

Rosas FE, Mediano PAM, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et al. (2020)
_"Reconciling emergences: An information-theoretic approach to identify causal emergence in
multivariate data"_. PLoS Comput Biol 16(12):e1008289. arXiv:2004.08220

Lizier JT (2014). _"JIDT: An information-theoretic toolkit for studying the dynamics of complex systems"_.
Frontiers in Robotics and AI 1:11. arXiv:1408.3270.
doi:10.3389/frobt.2014.00011.
