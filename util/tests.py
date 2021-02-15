#!/usr/bin/python

# to run, make sure you are in the root of the repo, and
#   python -m util.tests

from util.geometry import *
from util.plot import plot_vector

import numpy as np


def test_scalar_e(i, actual, expected) -> bool:
    if abs(actual - expected) > 1e-5:
        print(f"Test {i}: failed! Expected {expected}, got {actual}")

def test_vector_e(i, actual, expected) -> bool:
    if any([abs(a - e) > 1e-5 for (a, e) in zip(actual, expected)]):
        print(f"Test {i}: failed! Expected {expected}, got {actual}")

def test_np_array(i, actual, expected) -> bool:
    if not np.array_equal(actual, expected):
        print(f"Test {i}: failed! Expected {expected}, got {actual}")

# test vector to angle and angle to vector conversions
test_scalar_e(1, vec_to_ang([ 1,  0]),  0)
test_scalar_e(2, vec_to_ang([-1,  0]),  pi)
test_scalar_e(3, vec_to_ang([ 0,  1]),  pi/2)
test_scalar_e(4, vec_to_ang([ 0, -1]), -pi/2)
test_scalar_e(5, vec_to_ang([ 1,  1]),  pi/4)
test_scalar_e(6, vec_to_ang([ 1, -1]), -pi/4)
test_scalar_e(7, vec_to_ang([-1,  1]),  3*pi/4)
test_scalar_e(8, vec_to_ang([-1, -1]), -3*pi/4)

test_vector_e(9,  ang_to_vec(   0),   [ 1,  0])
test_vector_e(10, ang_to_vec(  pi),   [-1,  0])
test_vector_e(11, ang_to_vec( -pi),   [-1,  0])
test_vector_e(12, ang_to_vec(  pi/2), [ 0,  1])
test_vector_e(13, ang_to_vec( -pi/2), [ 0, -1])
test_vector_e(14, ang_to_vec(  pi/4), [ 1/sqrt(2), 1/sqrt(2)])
test_vector_e(15, ang_to_vec(3*pi/4), [-1/sqrt(2), 1/sqrt(2)])

test_scalar_e(16, vec_to_ang(ang_to_vec( pi/3)),  pi/3)
test_scalar_e(17, vec_to_ang(ang_to_vec(-pi/3)), -pi/3)

# test neighbours
X = np.array([ [4, 4], [2, 2], [2, 2.8], [2.5, 2.5], [3, 3], [4, 4.1] ])
test_vector_e(18, neighbours(1, X, 1,    'metric'),      [ 1, 2, 3 ])
test_vector_e(19, neighbours(1, X, 2,    'metric'),      [ 1, 2, 3, 4 ])
test_vector_e(20, neighbours(0, X, 0.11, 'metric'),      [ 0, 5 ])
test_vector_e(21, neighbours(0, X, 0.1,  'metric'),      [ 0 ])
test_vector_e(22, neighbours(0, X, 0.5,  'metric'),      [ 0, 5 ])
test_vector_e(23, neighbours(2, X, 0.5,  'metric'),      [ 2 ])
test_vector_e(24, neighbours(0, X, 2,    'topological'), [ 0, 5, 4 ])

# test average direction
test_scalar_e(25, average_angles([ [1], [-1.1] ]), np.average([ 1, -1.1]))
test_scalar_e(26, average_angles([ [2], [1] ]),    1.5)
test_scalar_e(27, average_angles([ [2], [-2] ]),   np.pi)
