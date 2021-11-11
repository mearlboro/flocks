# Library for simulating flocks


Implemented in Python and packaged with `pipenv`. Use

```
$ pipenv install -r requirements.txt
$ pipenv shell
$ python -m flock.main {model_name} --help
```

for information on simulation parameters, and run also with `python -m`.

Simulation results are stored in `out/txt`. If `--saveimg` param is set, plots
of the state of the simulation at each timestep are saved in `out/img`.

Use `./gif.sh` to turn all the images into animations of the model.

## Computing flock emergence

Uses code from Rosas FE, Mediano PAM, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et al. (2020)
_"Reconciling emergences: An information-theoretic approach to identify causal emergence in
multivariate data"_. PLoS Comput Biol 16(12):e1008289.

Uses `infodynamics.jar` binary of the Java Information Dynamics Toolkit (JIDT).
Lizier JT (2014). _"JIDT: An information-theoretic toolkit for studying the dynamics of complex systems"_. arXiv:1408.3270

To run an example emergence computation on random data

```
$ python analysis.emergence_calculator
```

