import dit
from dit.shannon import mutual_information as mi
from dit.distribution import BaseDistribution
from math import isclose
import numpy as np
from typing import Dict, List, Tuple, Union, Set

'''
Implement the theory of causal emergence in discrete information theory package
dit using mutual infomation
'''
def _vmi(dist: BaseDistribution) -> float:
    Xs = [ x[0] for x in dist.rvs[:-1]]
    V  = dist.rvs[-1]
    return mi(dist, Xs, V)

def _xvmi(dist: BaseDistribution) -> float:
    X = dist.rvs[:-1]
    V = dist.rvs[-1]
    sum_mi = sum( mi(dist, x, V) for x in X )
    return sum_mi

def _vxmi(dist: BaseDistribution) -> float:
    X = dist.rvs[:-1]
    V = dist.rvs[-1]
    sum_mi = sum( mi(dist, V, x) for x in X )
    return sum_mi

def _xmi(dist: BaseDistribution) -> float:
    X = dist.rvs[:-1]
    sum_mi = sum( mi(dist, xi, xj) for xi in X for xj in X )
    return sum_mi

def _xjmi(dist: BaseDistribution, xj: List[int]) -> float:
    X = dist.rvs[:-1]
    sum_mi = sum( mi(dist, xi, xj) for xi in X )
    return sum_mi



def mi_psi(dist: BaseDistribution) -> float:
    return _vmi(dist) - _xvmi(dist)

def mi_delta(dist: BaseDistribution) -> float:
    X = dist.rvs[:-1]
    V = dist.rvs[-1]
    return max( mi(dist, V, xj) - _xjmi(dist, xj) for xj in X )

def mi_gamma(dist: BaseDistribution) -> float:
    X = dist.rvs[:-1]
    V = dist.rvs[-1]
    return max( mi(dist, V, x) for x in X )

'''
Implement the theory of causal emergence in discrete information theory package
dit using the PID lattice expansion
'''

def count_singletons(atom: Tuple[Tuple[int, ...], ...]) -> int:
    '''
    Given a PID atom from the PID lattice, count how many singletons it contains
    e.g.

    {1}       = ((1,),)            => 1
    {1}{23}   = ((1,), (2, 3))     => 1
    {1}{2}{3} = ((1,), (2,), (3,)) => 3
    '''
    ls = [ len(item) for item in atom ]
    return ls.count(1)


def singletons(pid: "dit.pid.pid.BasePiD") -> List[Tuple[int, ...]]:
    '''
    Given a PID lattice produced by a dit.pid measure, return singleton atoms
    i.e. atoms of the form {1}, containing unique information, noted as the tuple
    containing only the element (0,) in dit
    '''
    sgls = list(pid._lattice.irreducibles())
    return sgls

def singletons_ascendants(
        pid: "dit.pid.pid.BasePID"
    ) -> Set[Tuple[Union[int, Tuple[int, ...]], ...]]:
    '''
    Given a PID lattice produced by a dit.pid measure, return all atoms that are
    ascendants of the singletons, i.e. in the top half of the lattice
    '''
    sets = [ pid._lattice.ascendants(atom) for atom in singletons(pid) ]
    atoms = set()
    for s in sets:
        atoms.update(s)
    return atoms

def singletons_descendants(
        pid: "dit.pid.pid.BasePID"
    ) -> Set[Tuple[Union[int, Tuple[int, ...]], ...]]:
    '''
    Given a PID lattice produced by a dit.pid measure, return all atoms that are
    descendants of the singletons, i.e. in the bottom half of the lattice
    '''
    sets = [ pid._lattice.descendants(atom) for atom in singletons(pid) ]
    atoms = set()
    for s in sets:
        atoms.update(s)
    return atoms

def n_singletons(
        pid: "dit.pid.pid.BasePID", n: int
    ) -> List[Tuple[Union[int, Tuple[int, ...]], ...]]:
    '''
    Given a PID lattice produced by a dit.pid measure, return all atoms that
    have a specified number of singletons
    '''
    return [ atom for atom in pid._lattice if count_singletons(atom) == n ]

def pid_psi(
        pid: "dit.pid.pid.BasePID"
    ) -> float:
    '''
    Given a PID lattice produced by a dit.pid measure, compute the Psi measure
    as the sum between all terms without singeltons and the sum of all terms
    with any singltetons, weighted by their singleton count
    '''
    n_src = len(pid._lattice.bottom)
    psi = 0.0

    for g in range(n_src):
        atoms = n_singletons(pid, g)
        psi += (1 - g) * sum( pid.get_pi(atom) for atom in atoms )

    return psi

