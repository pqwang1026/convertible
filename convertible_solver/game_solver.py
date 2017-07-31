import numpy as np
import logging
import matplotlib.pyplot as plt
import utils.distribution as dist
import pandas as pd
from convertible_solver.minor_solver import MinorStoppingModel
from convertible_solver.major_solver import MajorStoppingModel
from solver.discrete_stopping_solver import DiscreteStoppingSolver
import utils.distribution

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
    handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CCBModel:
    def __init__(self):
        self.r = 0.01
        self.c = 0.1
        self.tau_0 = 9

        self.T = 10

        self.e = 10
        self.p = 1000
        self.M = 1e6
        self.N = 1e4
        self.k = 1.2
        self.d = 1
        self.nu = 0
        self.sigma = 0.4
        self.sigma_R = 0.0005

        self.v_0 = 100
        self.R_0 = 0.025

        self.time_num_grids = 50
        self.state_num_grids = 50

    def get_initial_major_stopping_dist(self):
        return utils.distribution.Distribution([i * self.T / self.time_num_grids for i in range(1, self.time_num_grids + 1)], [1 / self.time_num_grids for _ in range(1, self.time_num_grids + 1)])

    def get_initial_minor_stopping_dist(self):
        return utils.distribution.Distribution([i * self.T / self.time_num_grids for i in range(1, self.time_num_grids + 1)], [1 / self.time_num_grids for _ in range(1, self.time_num_grids + 1)])

    def get_minor_model(self, major_stopping_dist, minor_stopping_dist):
        model = MinorStoppingModel()
        model.r = self.r
        model.c = self.c

        model.major_stopping_dist = major_stopping_dist
        model.T = self.T

        model.e = self.e
        model.p = self.p
        model.M = self.M
        model.N = self.N
        model.k = self.k
        model.d = self.d
        model.nu = self.nu
        model.sigma = self.sigma

        model.v_0 = self.v_0

        model.I = minor_stopping_dist

        model.time_num_grids = self.time_num_grids
        model.time_upper_bound = model.T
        model.time_lower_bound = 0

        model.state_num_grids = self.state_num_grids
        model.state_upper_bound = model.v_0 * 2
        model.state_lower_bound = model.v_0 / 2

        return model

    def get_major_model(self, minor_stopping_dist):
        model = MajorStoppingModel()
        model.r = self.r
        model.c = self.c
        model.T = self.T

        model.e = self.e
        model.p = self.p
        model.M = self.M
        model.N = self.N
        model.k = self.k
        model.d = self.d
        model.nu = self.nu
        model.sigma = self.sigma_R

        model.R_0 = self.R_0

        model.I = minor_stopping_dist

        model.time_num_grids = self.time_num_grids
        model.time_upper_bound = model.T
        model.time_lower_bound = 0

        model.state_num_grids = self.state_num_grids
        model.state_upper_bound = model.R_0 * 4
        model.state_lower_bound = 0

        return model


class GameSolver:
    def __init__(self, bond_model, major_stopping_dist, minor_stopping_dist, num_mc=5000, num_grids=400, num_max_iter=25, precision=0.05):
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

        self.major_model = self.bond_model.get_major_model(self.minor_stopping_dist)
        self.minor_model = self.bond_model.get_minor_model(self.major_stopping_dist, self.minor_stopping_dist)

        self.major_solver = DiscreteStoppingSolver(self.major_model)
        self.minor_solver = DiscreteStoppingSolver(self.minor_model)

        self.major_solver.solve()
        self.minor_solver.solve()

        self.minor_stopping_dist = self.minor_solver.estimate_stopping_distribution(self.bond_model.v_0, num_samples=int(1e3))
        self.major_stopping_dist = self.major_solver.estimate_stopping_distribution(self.bond_model.R_0, num_samples=int(1e3))

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
            break


if __name__ == '__main__':
    bond_model = CCBModel()

    major_stopping_dist = bond_model.get_initial_major_stopping_dist()
    minor_stopping_dist = bond_model.get_initial_minor_stopping_dist()

    solver = GameSolver(bond_model, major_stopping_dist, minor_stopping_dist)
    solver.solve()
    # solver.plot_equilibrium_strats()
