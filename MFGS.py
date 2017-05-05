import numpy as np
import logging
from test import MinorStoppingModel, MajorStoppingModel, OptimalStoppingSolver, ConvertibleModel
import matplotlib.pyplot as plt
import distribution as dist
import pandas as pd

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
    handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MFSGSolver:
    def __init__(self, bond_model, major_stopping_dist=None, minor_stopping_dist=None):
        self.minor_model = None
        self.major_model = None

        self.minor_stopping_dist = minor_stopping_dist
        self.major_stopping_dist = major_stopping_dist

        self.minor_stopping_dist_prev = minor_stopping_dist
        self.major_stopping_dist_prev = major_stopping_dist

        self.num_mc = int(1e3)  # number of monte carlo trials

        self.bond_model = bond_model

    def update(self):
        self.minor_stopping_dist_prev = self.minor_stopping_dist
        self.major_stopping_dist_prev = self.major_stopping_dist

        self.minor_model = MinorStoppingModel(self.bond_model, self.major_stopping_dist, self.minor_stopping_dist, )
        self.major_model = MajorStoppingModel(self.bond_model, self.minor_stopping_dist)

        upper_bound = 0.8
        lower_bound = 0
        num_grids = 100
        major_solver = OptimalStoppingSolver(self.major_model, upper_bound, lower_bound, num_grids, self.num_mc)

        upper_bound = 30
        lower_bound = 0
        num_grids = 100
        minor_solver = OptimalStoppingSolver(self.minor_model, upper_bound, lower_bound, num_grids, self.num_mc)

        major_solver.solve_full()
        minor_solver.solve_full()

        self.minor_stopping_dist = minor_solver.stopping_distribution
        self.major_stopping_dist = major_solver.stopping_distribution

    def plot_incremental_comparison(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        major_cdf = pd.Series(self.major_stopping_dist.cdf)
        major_cdf_prev = pd.Series(self.major_stopping_dist_prev.cdf)
        ax.step(major_cdf.index, major_cdf.data, where='post', color='b')
        ax.step(major_cdf_prev.index, major_cdf_prev.data, where='post', color='r', linestyle='--')

        ax = fig.add_subplot(2, 1, 2)
        minor_cdf = pd.Series(self.minor_stopping_dist.cdf)
        minor_cdf_prev = pd.Series(self.minor_stopping_dist_prev.cdf)
        ax.step(minor_cdf.index, minor_cdf.data, where='post', color='b')
        ax.step(minor_cdf_prev.index, minor_cdf_prev.data, where='post', color='r', linestyle='--')

        plt.show()

    @property
    def logger(self):
        return logger

    def solve(self):
        while True:
            self.update()
            self.plot_incremental_comparison()


if __name__ == '__main__':
    v0 = 3.0
    delta = 0.5
    nu = 0.04
    c = 0.03
    r = 0.02
    r0 = 0.04
    sigma0 = 0.001
    sigma = 0.05
    dividend = 0.04
    mat = 30
    bond_model = ConvertibleModel(v0, r, nu, c, dividend, sigma, delta, mat, r0, sigma0)

    major_stopping_dist = dist.DiscreteDistribution(np.ones(mat + 1) / (mat + 1))
    minor_stopping_dist = dist.DiscreteDistribution(np.ones(mat + 2) / (mat + 2))

    solver = MFSGSolver(bond_model=bond_model, major_stopping_dist=major_stopping_dist, minor_stopping_dist=minor_stopping_dist)
    solver.solve()
