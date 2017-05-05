from OptimalStoppingSolver import ConvertibleModel, OptimalStoppingSolver
from MFGS import MeanFieldGameSolver
import numpy as np


def main(argv):
    v0 = 3.0
    delta = 0.5
    nu = 0.03
    tau0 = 10
    c = 0.03
    k = 1.2
    r = 0.02
    sigma = 0.03
    mat = 30
    mu = np.linspace(0, 1, 11)
    model = ConvertibleModel(v0, tau0, r, nu, c, k, sigma, delta, mu, mat)
    mfg_solver = MeanFieldGameSolver(model=model, grid_size=1000, monte_carlo=10000, precision=0.01, max_iter=50)
    mfg_solver.solve_full()
