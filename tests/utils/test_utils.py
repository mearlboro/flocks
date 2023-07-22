#!/usr/bin/python
from math import pi, sqrt

import matplotlib.cm as cm
import numpy as np
import pytest
from hamcrest import assert_that, close_to, is_, equal_to
from matplotlib import pyplot as plt

from pyflocks.util.geometry import vec_to_ang, ang_to_vec, neighbours, EnumNeighbours, EnumBounds, average_angles, \
    sum_vec_ang, centre_of_mass, periodic_diff, ang_mod, bearing_to
from pyflocks.util.plot import plot_vector

# to run, make sure you are in the root of the repo, and
#   python -m util.tests


# test angle sums and averages
n = 10
v = 0.1
V = np.ones((n, 1)) * v
A = np.random.uniform(-np.pi, np.pi, size=(n, 1))

# test cmass and distances for periodic bounds
l = 10
X = np.array([[1, 1], [9, 9], [1, 2]])
X1 = np.array([[1, 1], [-1, -1], [1, 2]])
X2 = np.array([[0, 0], [0, 9.9], [9.9, 0], [9.9, 9.9]])


@pytest.mark.parametrize("i,actual,expected", [
    1, vec_to_ang([1, 0]), 0,
    2, vec_to_ang([-1, 0]), pi,
    3, vec_to_ang([0, 1]), pi / 2,
    4, vec_to_ang([0, -1]), -pi / 2,
    5, vec_to_ang([1, 1]), pi / 4,
    6, vec_to_ang([1, -1]), -pi / 4,
    7, vec_to_ang([-1, 1]), 3 * pi / 4,
    8, vec_to_ang([-1, -1]), -3 * pi / 4,
    16, vec_to_ang(ang_to_vec(pi / 3)), pi / 3,
    17, vec_to_ang(ang_to_vec(-pi / 3)), -pi / 3,
    18, vec_to_ang(ang_to_vec(4 * pi - pi / 3)), -pi / 3,
    19, vec_to_ang(ang_to_vec(-4 * pi + pi / 3)), pi / 3,
    29, average_angles([[1], [-1.1]]), np.average([1, -1.1]),
    30, average_angles([[2], [1]]), 1.5,
    31, average_angles([[2], [-2]]), np.pi,
    33, average_angles(A), vec_to_ang(np.mean([ang_to_vec(a) * v for a in A], axis=0)),
    42, ang_mod(1), 1,
    43, ang_mod(-1), -1,
    44, ang_mod(pi), pi,
    45, ang_mod(-pi), -pi,
    45, ang_mod(-pi + 1), -pi + 1,
    40, periodic_diff(X[0], X[1], l, True), 2 * np.sqrt(2),
    41, periodic_diff(X[0], X[2], l, True), 1,
    46, bearing_to(np.deg2rad(45), [1, 1]), 0,
    47, bearing_to(np.deg2rad(45), [0, 1]), np.deg2rad(-135),
])
def test_scalar_e(i, actual, expected) -> None:
    assert_that(actual, close_to(expected, 1e-5), f"Test {i}: failed! Expected {expected}, got {actual}")


@pytest.mark.parametrize('i,actual,expected', [
    9, ang_to_vec(0), [1, 0],
    10, ang_to_vec(pi), [-1, 0],
    11, ang_to_vec(-pi), [-1, 0],
    12, ang_to_vec(pi / 2), [0, 1],
    13, ang_to_vec(-pi / 2), [0, -1],
    14, ang_to_vec(pi / 4), [1 / sqrt(2), 1 / sqrt(2)],
    15, ang_to_vec(3 * pi / 4), [-1 / sqrt(2), 1 / sqrt(2)],
    20, neighbours(1, X, 1, EnumNeighbours.METRIC, EnumBounds.REFLECTIVE), [1, 2, 3],
    21, neighbours(0, X, 0.11, EnumNeighbours.METRIC, EnumBounds.REFLECTIVE), [0, 5],
    22, neighbours(0, X, 0.1, EnumNeighbours.METRIC, EnumBounds.REFLECTIVE), [0],
    23, neighbours(1, X, 3, EnumNeighbours.TOPOLOGICAL, EnumBounds.REFLECTIVE), [1, 3, 2, 4],
    24, neighbours(0, X, 3, EnumNeighbours.TOPOLOGICAL, EnumBounds.REFLECTIVE), [0, 5, 4, 3],
    25, neighbours(1, X, 1, EnumNeighbours.METRIC, EnumBounds.PERIODIC, 5), [1, 2, 3],
    26, neighbours(0, X, 1, EnumNeighbours.METRIC, EnumBounds.PERIODIC, 5), [0, 5],
    27, neighbours(1, X, 5, EnumNeighbours.TOPOLOGICAL, EnumBounds.PERIODIC, 5), [1, 3, 2, 4, 6, 0],
    28, neighbours(0, X, 5, EnumNeighbours.TOPOLOGICAL, EnumBounds.PERIODIC, 5), [0, 5, 4, 7, 8, 6],
    32, sum_vec_ang(A, V), np.mean([ang_to_vec(a) * v for a in A], axis=0),
    34, centre_of_mass(X, l, EnumBounds.PERIODIC), centre_of_mass(X1, l, EnumBounds.PERIODIC),
    35, centre_of_mass(X2, l, EnumBounds.PERIODIC), [9.95, 9.95],
    36, centre_of_mass(X2, l, EnumBounds.REFLECTIVE), [4.95, 4.95],
    37, periodic_diff(X[0], X[1], l), np.array([2, 2]),
    38, periodic_diff(X[0], X[2], l), np.array([0, -1]),
    39, periodic_diff(X[2], X[0], l), np.array([0, 1]),
])
def test_vector_e(i, actual, expected) -> None:
    assert_that(len(actual), is_(equal_to(len(expected))))
    assert_that(any(abs(a - e) > 1e-5 for a, e, in zip(actual, expected)), is_(equal_to(False)),
                f"Test {i}: failed! Expected {expected}, got {actual}")


def test_np_array(i, actual, expected) -> None:
    if not np.array_equal(actual, expected):
        print(f"Test {i}: failed! Expected {expected}, got {actual}")


# show graphically the angles and plotting function
col = cm.rainbow(np.linspace(0, 1, 30))
phi = np.linspace(-2 * pi, pi, 30)
vec = list(map(ang_to_vec, phi))
for i in range(len(phi)):
    plot_vector([0, 0], vec[i], col[i])
plt.show()
phi = list(map(ang_mod, phi))
vec = list(map(ang_to_vec, phi))
for i in range(len(phi)):
    plot_vector([0, 0], vec[i], col[i])
plt.show()
