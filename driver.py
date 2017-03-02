from OptimalStoppingSolver import ConvertibleModel, OptimalStoppingSolver
from MFGS import MeanFieldGameSolver
import numpy as np

v0 = 3.0
delta = 0.5
nu = 0.03
tau0 = 30
c = 0.02
k = 1.2
r = 0.03
sigma = 0.05
mat = 30
mu = np.zeros(mat + 1)
mu[30] = 1.0
mu[29] = 1.0
mu[28] = 1.0
mu[27] = 1.0
model = ConvertibleModel(v0, tau0, r, nu, c, k, sigma, delta, mu, mat)
solver = OptimalStoppingSolver(model, 500, 10000)
mfg_solver = MeanFieldGameSolver(solver, mu, 0.1, 100)
mfg_solver.solve()

