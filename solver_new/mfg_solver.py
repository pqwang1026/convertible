import numpy as np
import logging
from solver_new.stopping_solver import MinorStoppingModel, MajorStoppingModel, OptimalStoppingSolver, ConvertibleModel
import matplotlib.pyplot as plt
import utils.distribution as dist
import pandas as pd

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
    handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MFSGSolver:
    def __init__(self, bond_model, major_stopping_dist, minor_stopping_dist, num_mc=5000, num_grids=400,
                 num_max_iter=25, precision=0.05):
        self.minor_model = None
        self.major_model = None

        self.minor_solver = None
        self.major_solver = None

        self.minor_stopping_dist = minor_stopping_dist
        self.major_stopping_dist = major_stopping_dist

        self.minor_stopping_dist_prev = minor_stopping_dist
        self.major_stopping_dist_prev = major_stopping_dist

        self.num_mc = num_mc  # number of monte carlo trials

        self.num_grids = num_grids  # number of grids for optimal stopping

        self.bond_model = bond_model

        self.num_max_iter = num_max_iter
        self.precision = precision
        self.error = 10000.00
        self.num_iter = 0
        self.continue_flag = 1

    def update(self):
        self.minor_stopping_dist_prev = self.minor_stopping_dist
        self.major_stopping_dist_prev = self.major_stopping_dist

        self.minor_model = MinorStoppingModel(self.bond_model, self.major_stopping_dist, self.minor_stopping_dist)
        self.major_model = MajorStoppingModel(self.bond_model, self.minor_stopping_dist)

        self.major_solver = OptimalStoppingSolver(self.major_model, self.num_grids, self.num_mc)
        self.minor_solver = OptimalStoppingSolver(self.minor_model, self.num_grids, self.num_mc)

        self.major_solver.solve_full()
        self.minor_solver.solve_full()

        self.minor_stopping_dist = self.minor_solver.stopping_distribution
        self.major_stopping_dist = self.major_solver.stopping_distribution

        self.num_iter = self.num_iter + 1
        self.error = np.linalg.norm(self.major_stopping_dist.pdf - self.major_stopping_dist_prev.pdf) + \
                     np.linalg.norm(self.minor_stopping_dist.pdf - self.minor_stopping_dist_prev.pdf)

        self.logger.info('Error = {0}'.format(self.error))

        if self.num_iter == self.num_max_iter or self.error <= self.precision:
            self.continue_flag = 0
            if self.num_iter == self.num_max_iter:
                self.logger.info('Warning: Maximum iteration reached...Give up...')
            if self.error <= self.precision:
                self.logger.info('Converge!')
                self.plot_incremental_comparison()

    def plot_incremental_comparison(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        major_cdf = pd.Series(self.major_stopping_dist.cdf)
        major_cdf_prev = pd.Series(self.major_stopping_dist_prev.cdf)
        ax.step(major_cdf.index, major_cdf.data, where='post', color='b')
        ax.step(major_cdf_prev.index, major_cdf_prev.data, where='post', color='r', linestyle='--')
        ax.set_title('major')

        ax = fig.add_subplot(2, 1, 2)
        minor_cdf = pd.Series(self.minor_stopping_dist.cdf)
        minor_cdf_prev = pd.Series(self.minor_stopping_dist_prev.cdf)
        ax.step(minor_cdf.index, minor_cdf.data, where='post', color='b')
        ax.step(minor_cdf_prev.index, minor_cdf_prev.data, where='post', color='r', linestyle='--')
        ax.set_title('minor')

        plt.show()

    def plot_equilibrium_strats(self):
        self.minor_solver.plot_stop_region()
        self.major_solver.plot_stop_region()

    def get_convergence_error(self):
        major_error = np.linalg.norm(self.major_stopping_dist.pdf - self.major_stopping_dist_prev.pdf)
        minor_error = np.linalg.norm(self.minor_stopping_dist.pdf - self.minor_stopping_dist_prev.pdf)
        return major_error + minor_error

    @property
    def logger(self):
        return logger

    def solve(self):
        error = 10000.0
        num_iterations = 0
        while self.continue_flag == 1:
            self.update()


if __name__ == '__main__':
    v0 = 6.0
    delta = 0.1
    nu = 0.02
    c = 0.05
    r = 0.01
    r0 = 0.04
    sigma = 0.1
    sigma0 = 0.01
    dividend = 0.01
    mat = 15
    bond_model = ConvertibleModel(v0, r, nu, c, dividend, sigma, delta, mat, r0, sigma0)

    major_stopping_dist = dist.DiscreteDistribution(np.ones(mat + 1) / (mat + 1))
    minor_stopping_dist = dist.DiscreteDistribution(np.ones(mat + 2) / (mat + 2))

    solver = MFSGSolver(bond_model=bond_model, major_stopping_dist=major_stopping_dist, minor_stopping_dist=minor_stopping_dist)
    solver.solve()
    # solver.plot_equilibrium_strats()
