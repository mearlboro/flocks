#!/usr/bin/python

# to run, make sure you are in the root of the repo, and
#   python -m util.tests

from util.geometry import *
from util.plot import plot_vector
import numpy as np
import matplotlib.pyplot as plt


def test_scalar(actual, expected) -> bool:
    if abs(actual - expected) > 1e-5:
        print(f"Test failed! Expected {expected}, got {actual}")

def test_vector(actual, expected) -> bool:
    if any([abs(a - e) > 1e-5 for (a, e) in zip(actual, expected)]):
        print(f"Test failed! Expected {expected}, got {actual}")


# test vector to angle and angle to vector conversions
test_scalar(vec_to_ang([ 1,  0]),  0)
test_scalar(vec_to_ang([-1,  0]),  pi)
test_scalar(vec_to_ang([ 0,  1]),  pi/2)
test_scalar(vec_to_ang([ 0, -1]), -pi/2)
test_scalar(vec_to_ang([ 1,  1]),  pi/4)
test_scalar(vec_to_ang([ 1, -1]), -pi/4)
test_scalar(vec_to_ang([-1,  1]),  3*pi/4)
test_scalar(vec_to_ang([-1, -1]), -3*pi/4)

test_vector(ang_to_vec(   0),   [ 1,  0])
test_vector(ang_to_vec(  pi),   [-1,  0])
test_vector(ang_to_vec( -pi),   [-1,  0])
test_vector(ang_to_vec(  pi/2), [ 0,  1])
test_vector(ang_to_vec( -pi/2), [ 0, -1])
test_vector(ang_to_vec(  pi/4), [ 1/sqrt(2), 1/sqrt(2)])
test_vector(ang_to_vec(3*pi/4), [-1/sqrt(2), 1/sqrt(2)])

test_scalar(vec_to_ang(ang_to_vec( pi/3)),  pi/3)
test_scalar(vec_to_ang(ang_to_vec(-pi/3)), -pi/3)
