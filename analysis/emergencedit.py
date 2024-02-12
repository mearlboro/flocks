import dit
from dit.shannon import mutual_information as mi
from dit.distribution import BaseDistribution
import itertools as it
import math
import numpy as np

from typing import Callable, Dict, List, Tuple, Union, Set

'''
Implement the theory of causal emergence in discrete information theory package
dit using mutual infomation
'''

def _vmi(dist: BaseDistribution) -> float:
    '''
    Compute mutual information of the whole system (e.g. all source vars,
    jointly) with the target
    '''
    Xs = [ x[0] for x in dist.rvs[:-1]]
    V  = dist.rvs[-1]
    return mi(dist, Xs, V)

def _xvmi_sum(dist: BaseDistribution) -> float:
    '''
    Compute the sum of mutual information between each individual source and
    target
    '''
    X = dist.rvs[:-1]
    V = dist.rvs[-1]
    sum_mi = sum( mi(dist, x, V) for x in X )
    return sum_mi

def _xvmi_int(
        dist: BaseDistribution, X: List[int],
        min_func: Callable[[float, float], float] = min
    ) -> float:
    '''
    Compute intersection information between sources in the list X and target.
    By default uses minimum mutual information
    '''
    V = dist.rvs[-1]
    int_mi = min( mi(dist, x, V) for x in X )
    return int_mi

def _psi_coef(n: int, q: int, r: int) -> int:
    '''
    Compute coefficient for the double-counting redundancy correction
    '''
    if r == n - q + 1:
        return n - q
    else:
        return r - 1 - sum( _psi_coef(n, q, s) * math.comb(r, s)
                            for s in range(n - q + 1, r))

def _xvmi_corr(dist: BaseDistribution, q: int) -> float:
    '''
    Expand PID lattice using intersection information between groups of sources
    and targets according to order of redundancy correction
    '''
    V = dist.rvs[-1]
    X = dist.rvs[:-1]
    n = len(X)
    corr = 0.0
    for r in range(n - q + 1, n + 1):
        X_sets = list(it.combinations(X, r))
        coef = _psi_coef(n, q, r)
        corr += sum(coef * _xvmi_int(dist, x) for x in X_sets)
    return corr

def _xjmi(dist: BaseDistribution, xj: List[int]) -> float:
    '''
    Compute the sum of mutual information between a given source xj and all
    sources
    '''
    X = dist.rvs[:-1]
    sum_mi = sum( mi(dist, xi, xj) for xi in X )
    return sum_mi


def mi_psi(dist: BaseDistribution, q: int = 0) -> float:
    '''
    Compute the Psi measure as the difference between how the sources jointly
    and individually predict the target
    '''
    return _vmi(dist) - _xvmi_sum(dist) + _xvmi_corr(dist, q)

def mi_delta(dist: BaseDistribution) -> float:
    '''
    Compute the downward causation Delta measure as the maximum difference
    between the mutual information between the target and each source and the
    sum of mutual information between that source and all other sources
    '''
    X = dist.rvs[:-1]
    V = dist.rvs[-1]
    return max( mi(dist, V, xj) - _xjmi(dist, xj) for xj in X )

def mi_gamma(dist: BaseDistribution) -> float:
    '''
    Compute the causal decoupling Gamma measure as the maximum mutual info
    between the target and each individual source
    '''
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
        pid: "dit.pid.pid.BasePID", q: int = 0
    ) -> float:
    '''
    Given a PID lattice produced by a dit.pid measure, compute the Psi measure
    as the sum between all terms without singeltons and the sum of all terms
    with any singltetons, weighted by their singleton count
    '''
    n_src = len(pid._lattice.bottom)
    psi = 0.0

    for g in range(n_src + 1 - q):
        atoms = n_singletons(pid, g)
        psi += (1 - g) * sum( pid.get_pi(atom) for atom in atoms )

    return psi


'''
Test the theory against multiple cases
'''
if __name__ == "__main__":
    stats = []
    for n_var in range(3, 5):
        for q in range(0, n_var):
            diffs = []
            for _ in range(30):
                d = dit.random_distribution(n_var, 2)
                i = dit.pid.PID_MMI(d)
                ppid = pid_psi(i, q = q)
                pmi  = mi_psi(d, q = q)
                diffs.append(np.abs(ppid - pmi))
            m = np.mean(diffs)
            s = np.std(diffs)
            print(f"n={n_var}\tq={q}\t{m}\t{s}")
