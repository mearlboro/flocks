# Library for simulating flocks

Implemented in Python and packaged with `pipenv`. Use

```
$ pipenv install -r requirements.txt
$ pipenv shell
$ python -m flock.main {model_name} --help
```

for information on simulation parameters, and run also with `python -m`. Default
parameters are specified in `main.py`.

Simulation results are stored in `out/txt`. If `--saveimg` param is set, plots
of the state of the simulation at each timestep are saved in `out/img`.

For the purposes of exploring a variable that describes the whole system, the
vector sum of particle velocities or the trajctories can be included in the images.

Use `./gif.sh` to turn all the images into animations of the model.

## Models
Currently the Standard Vicsek Model (SVM) is implemented reliably, plus topological
interactions and reflective boundaries.

A variant of the model is also implemented where each particle is a Kuramoto
oscillator. This has not been fully tested to full scrutiny.

Reynolds flocking model is also in progress, but it does NOT behave correctly.

## Examples
```
$ python -m flock.main vicsek -n 100 -l 5 --bounds PERIODIC --neighbours METRIC \
        --saveimg
$ cd out
$ ./gif.sh
```
![](/out/gif/Vicsek_periodic_topological_eta0.4_v0.1_r1.0_rho4.0.gif)

```
$ python -m flock.main vicsek -e 1 -n 20 -l 2 --bounds PERIODIC --neighbours METRIC \
        --saveimg --trajectories
$ cd out
$ ./gif.sh
```
![](/out/gif/Vicsek_periodic_metric_eta1.0_v0.1_r1.0_rho5.0_traj.gif)

```
$ python -m flock.main vicsek e 0.7 -n 20 -l 2 --bounds PERIODIC --neighbours METRIC \
        --saveimg --sumvec
$ cd out
$ ./gif.sh
```
![](/out/gif/Vicsek_reflective_topological_eta0.7_v0.1_r3.0_rho5.0_sumvec.gif)

# Analysis and plotting

Code located in `analysis/` is for computing statistics and emergence measures on
the exported simulation trajectories in `out/txt`.

### System statistics

The first step in analysis happens in `stats.py` which contains three main functions:
* `process_space`, which takes the 3D numpy array of trajectories and the size and topology of the simulation space and computes centre of mass, the vector of distances of each particle from centre of mass, as well as the the average and standard deviation of the distances for each time step, returned in a dictionary
* `process_angles`, which takes the 2D numpy array of the angular velocity of each particle in the system, as well as the absolute velocity of all particles, and computes the average angle, the standard deviation from that angle, as well as the Vicsek order parameter the absolute average velocity for each time step, returned in a dictionary
* `autocorellation`, which takes any type of numpy array (either produced directly by the system, e.g. a trajectory, or a system variable as produced by `process_space` or `process_angles`) and a maximum window size, and computes the average Pearson corellation function between the value at time `t` and the value at time `t+w` for all window sizes `w` smaller than the maximum window size

### Analysing one simulation
Results produced by the `stats.py` functions above can be plotted with `plot_stats.py`.
This file takes all the simulation outputs in a given folder and produces plots of the statistics for each of the simulations, in simulation-specific subfolders.
The default output folder for plots is located in `/out/plt`.
The x-axis is always time (or, in the case of autocorellations, the window size).

```
$ python -m analysis.plot_stats --path {model_path} --model {model_name} --out {output_path}
```

For consistency checking, the trajectories of the particles for that simulation are also plotted.


### Analysing multiple simulated realisations of the same model

For a number of simulations using the same parameters, aggregated plots can be produced using `aggregate_plot.py`. The same `stats.py` functions are used but average and standard deviation accros runs are computed with numpy. The parameters by which the statistics should be aggregated are passed as command line flags. Use

```
$ python -m analysis.plot_aggregate --path {model_path} --model {model_name} --out {output_path} -a {param1} -a {param2} [-a {param3}]
```

## Examples

Inspired by the order parameter analysis done by Vicsek et. al, one could plot the absolute average velocity, averaged over time and accross the multiple experiment, against the noise parameter (eta), producing a plot for each different density (rho).
To produce the plot below, after 50 simulations were created with the same `n`, `rho`, and a range of `eta`:

```
$ python -m analysis.plot_aggregate --path out/txt/ --model 'Vicsek_periodic_metric' -a rho -a eta
```

![](/out/plt/Vicsek_periodic_metric_rho2.5_eta_vs_avg_abs_vel.png)

In case 3 aggregate parameters are passed, one plot will be made for each value of the first, plotting statistics against values of the second, for each value of the third e.g

```
$ python -m analysis.plot_aggregate --path out/txt/ --model 'Vicsek_reflective_topological' -a rho -a eta -a r
```

![](/out/plt/Vicsek_reflective_topological_rho2.5_eta_r_vs_avg_abs_vel.png)

# Computing flock emergence

Uses code from Rosas FE, Mediano PAM, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et al. (2020)
_"Reconciling emergences: An information-theoretic approach to identify causal emergence in
multivariate data"_. PLoS Comput Biol 16(12):e1008289.

Uses `infodynamics.jar` binary of the Java Information Dynamics Toolkit (JIDT).
Lizier JT (2014). _"JIDT: An information-theoretic toolkit for studying the dynamics of complex systems"_. arXiv:1408.3270

To run an example emergence computation on random data

```
$ python -m analysis.emergence
```

or on an existing model's trajectories, say `Vicsek_periodic_metric_eta0.1_v0.1_r1.0_rho1.0`

```
$ python -m analysis.emergence --model out/img/Vicsek_periodic_metric_eta0.1_v0.1_r1.0_rho1.0
```

