#!/usr/bin/python

# to run, make sure you are in the root of the repo, and
#   python -m util.tests

from util.geometry import *
from util.plot import *

import numpy as np
import matplotlib.cm as cm

def test_scalar_e(i, actual, expected) -> None:
    if abs(actual - expected) > 1e-5:
        print(f"Test {i}: failed! Expected {expected}, got {actual}")

def test_vector_e(i, actual, expected) -> None:
    if any([abs(a - e) > 1e-5 for (a, e) in zip(actual, expected)]):
        print(f"Test {i}: failed! Expected {expected}, got {actual}")

def test_np_array(i, actual, expected) -> None:
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
test_scalar_e(18, vec_to_ang(ang_to_vec(4*pi-pi/3)), -pi/3)
test_scalar_e(19, vec_to_ang(ang_to_vec(-4*pi+pi/3)), pi/3)

# test neighbours
X = np.array([ [4, 4], [2, 2], [2, 2.8], [2.5, 2.5], [3, 3], [4, 4.1] ])
test_vector_e(20, neighbours(1, X, 1,    EnumNeighbours.METRIC),      [ 1, 2, 3 ])
test_vector_e(21, neighbours(1, X, 2,    EnumNeighbours.METRIC),      [ 1, 2, 3, 4 ])
test_vector_e(22, neighbours(0, X, 0.11, EnumNeighbours.METRIC),      [ 0, 5 ])
test_vector_e(23, neighbours(0, X, 0.1,  EnumNeighbours.METRIC),      [ 0 ])
test_vector_e(24, neighbours(0, X, 0.5,  EnumNeighbours.METRIC),      [ 0, 5 ])
test_vector_e(25, neighbours(2, X, 0.5,  EnumNeighbours.METRIC),      [ 2 ])
test_vector_e(26, neighbours(0, X, 2,    EnumNeighbours.TOPOLOGICAL), [ 0, 5, 4 ])

# test average direction
test_scalar_e(27, average_angles([ [1], [-1.1] ]), np.average([ 1, -1.1]))
test_scalar_e(28, average_angles([ [2], [1] ]),    1.5)
test_scalar_e(29, average_angles([ [2], [-2] ]),   np.pi)

# test angle sums and averages
n = 10
v = 0.1
V = np.ones((n, 1)) * v
A = np.random.uniform(-np.pi, np.pi, size=(n, 1))

test_vector_e(30, sum_vec_ang(A, V),
                  np.mean([ ang_to_vec(a) * v for a in A], axis = 0) * n )
test_scalar_e(31, average_angles(A),
                  vec_to_ang(np.mean([ ang_to_vec(a) * v for a in A], axis = 0)))

# test cmass and distances for periodic bounds
l = 10
X  = np.array([[1, 1], [9,  9], [1, 2]])
X1 = np.array([[1, 1], [-1,-1], [1, 2]])
X2 = np.array([[0, 0], [0, 9.9], [9.9, 0], [9.9, 9.9]])

test_vector_e(32, centre_of_mass(X,  l, EnumBounds.PERIODIC), centre_of_mass(X1, l, EnumBounds.PERIODIC))
test_vector_e(33, centre_of_mass(X2, l, EnumBounds.PERIODIC),   [9.95, 9.95])
test_vector_e(34, centre_of_mass(X2, l, EnumBounds.REFLECTIVE), [4.95, 4.95])

test_vector_e(35, periodic_diff(X[0], X[1], l), np.array([2,  2]))
test_vector_e(36, periodic_diff(X[0], X[2], l), np.array([0, -1]))
test_vector_e(37, periodic_diff(X[2], X[0], l), np.array([0,  1]))
test_scalar_e(38, periodic_diff(X[0], X[1], l, True), 2 * np.sqrt(2))
test_scalar_e(39, periodic_diff(X[0], X[2], l, True), 1)

# test angle mod
test_scalar_e(42, ang_mod(1),      1)
test_scalar_e(43, ang_mod(-1),    -1)
test_scalar_e(44, ang_mod(pi),     pi)
test_scalar_e(45, ang_mod(-pi),   -pi)
test_scalar_e(45, ang_mod(-pi+1), -pi+1)

# test angle to point difference
test_scalar_e(40, bearing_to(np.deg2rad(45), [1, 1]), 0)
test_scalar_e(41, bearing_to(np.deg2rad(45), [0, 1]), np.deg2rad(-135))

print("If nothing so far was printed, it means your tests all pass! Good job.")

# show graphically the angles and plotting function
col = cm.rainbow(np.linspace(0, 1, 30))
phi = np.linspace(-2*pi, pi, 30)
vec = list(map(ang_to_vec, phi))
for i in range(len(phi)):
    plot_vector([0,0], vec[i], col[i])
plt.show()
phi = list(map(ang_mod, phi))
vec = list(map(ang_to_vec, phi))
for i in range(len(phi)):
    plot_vector([0,0], vec[i], col[i])
plt.show()
