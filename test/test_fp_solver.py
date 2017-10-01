import numpy as np
import utils.simplex as splx
import logging
from solver.fp_solver import *


def f(s: splx.SimplexPoint):
    """
    For a given 2-simplex point (x0, x1, x2), this function gives (sqrt(x1), 1-sqrt(x1), 0).
    """
    if s.n != 2:
        raise RuntimeError('dimension does not match!')
    return splx.SimplexPoint([np.sqrt(s[1]), 1 - np.sqrt(s[1]), 0])


def g(s: splx.SimplexPoint):
    """
    This is a rotation.
    """
    if s.n != 2:
        raise RuntimeError('dimension does not match!')
    return splx.SimplexPoint([s[1], s[2], s[0]])


func = f

# solver = IterataiveSolver(func, n=2)
# print(solver.solve())

solver = KakutaniSolver(func, n=2)
# print(solver.solve(k=4))

solver.solve_traversal(k=4)
