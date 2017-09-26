import numpy as np
import logging
import matplotlib.pyplot as plt
import utils.distribution as dist
import pandas as pd
from convertible_solver.minor_solver import MinorStoppingModel
from convertible_solver.major_solver import MajorStoppingModel
from solver.discrete_stopping_solver import DiscreteStoppingSolver
import utils.distribution

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CCBModel:
    def __init__(self):
        self.r = 0.01
        self.c = 0.02

        self.T = 10

        self.e = 16
        self.p = 1000
        self.M = 1e6
        self.N = 5e4
        self.k = 1
        self.d = 7
        self.nu = 0

        self.sigma_R = 0.003
        self.sigma = 0.2
        self.R_lt = 0.03
        self.theta = 0.1

        self.v_0 = 100
        self.R_0 = 0.03

        self.lambd = 1

        self.time_num_grids = 10
        self.state_num_grids = 100

    @classmethod
    def get_model(cls, r, c, T, e, p, M, N, k, d, nu, sigma, sigma_R, R_lt, theta, v_0, R_0, lambd, time_num_grids, state_num_grids):
        ret = cls()

        ret.r = r
        ret.c = c

        ret.T = T

        ret.e = e
        ret.p = p
        ret.M = M
        ret.N = N
        ret.k = k
        ret.d = d
        ret.nu = nu
        ret.sigma = sigma

        ret.sigma_R = sigma_R
        ret.R_lt = R_lt
        ret.theta = 0.1

        ret.v_0 = v_0
        ret.R_0 = R_0

        ret.lambd = lambd

        ret.time_num_grids = time_num_grids
        ret.state_num_grids = state_num_grids

        return ret

    def get_initial_major_stopping_dist(self):
        return utils.distribution.Distribution([i * self.T / self.time_num_grids for i in range(0, self.time_num_grids + 1)],
                                               [1 / (self.time_num_grids + 1) for _ in range(0, self.time_num_grids + 1)])

    def get_random_initial_dist(self):
        return utils.distribution.Distribution([i * self.T / self.time_num_grids for i in range(0, self.time_num_grids + 1)],
                                               utils.distribution.generate_simplex_sample(self.time_num_grids))

    def get_initial_minor_stopping_dist(self):
        return utils.distribution.Distribution([i * self.T / self.time_num_grids for i in range(0, self.time_num_grids + 1)],
                                               [1 / (self.time_num_grids + 1) for _ in range(0, self.time_num_grids + 1)])

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

        model.lambd = self.lambd

        model.time_num_grids = self.time_num_grids
        model.state_num_grids = self.state_num_grids

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
        model.theta = self.theta
        model.R_lt = self.R_lt

        model.I = minor_stopping_dist

        model.time_num_grids = self.time_num_grids
        model.state_num_grids = self.state_num_grids

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

        self.major_stopping_dist = self.major_solver.estimate_stopping_distribution(self.bond_model.R_0, num_samples=int(1e3))
        self.minor_stopping_dist = self.minor_solver.estimate_stopping_distribution(self.bond_model.v_0, num_samples=int(1e3))

        self.error = np.linalg.norm(self.major_stopping_dist.pdf.values - self.major_stopping_dist_prev.pdf.values) + \
                     np.linalg.norm(self.minor_stopping_dist.pdf.values - self.minor_stopping_dist_prev.pdf.values)

        self.logger.info('Error = {0}'.format(self.error))
        return self.error

    def update_(self):
        """
        Assume that self.minor_stopping_dist is well-defined, first use this measure to solve the major problem, record the major stopping distribution and plug it into minor's problem.
        Finally record minor stopping dist, and update error.
        """
        self.minor_stopping_dist_prev = self.minor_stopping_dist
        self.major_stopping_dist_prev = self.major_stopping_dist

        self.major_model = self.bond_model.get_major_model(self.minor_stopping_dist_prev)
        self.major_solver = DiscreteStoppingSolver(self.major_model)
        self.major_solver.solve()

        self.major_stopping_dist = self.major_solver.estimate_stopping_distribution(self.bond_model.R_0, num_samples=int(self.num_mc))

        self.minor_model = self.bond_model.get_minor_model(self.major_stopping_dist, self.minor_stopping_dist_prev)
        self.minor_solver = DiscreteStoppingSolver(self.minor_model)
        self.minor_solver.solve()
        self.minor_stopping_dist = self.minor_solver.estimate_stopping_distribution(self.bond_model.v_0, num_samples=int(self.num_mc))

        self.error = np.linalg.norm(self.minor_stopping_dist.pdf.values - self.minor_stopping_dist_prev.pdf.values)
        self.logger.info('After the update, the norm of I moved {0}'.format(self.error))
        return self.error

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

    def plot_major_incremental_comparison(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        major_cdf = pd.Series(self.major_stopping_dist.cdf)
        major_cdf_prev = pd.Series(self.major_stopping_dist_prev.cdf)
        ax.step(major_cdf.index, major_cdf.data, where='post', color='b')
        ax.step(major_cdf_prev.index, major_cdf_prev.data, where='post', color='r', linestyle='--')
        ax.set_title('major')

    def plot_minor_incremental_comparison(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        minor_cdf = pd.Series(self.minor_stopping_dist.cdf)
        minor_cdf_prev = pd.Series(self.minor_stopping_dist_prev.cdf)
        ax.step(minor_cdf.index, minor_cdf.data, where='post', color='b')
        ax.step(minor_cdf_prev.index, minor_cdf_prev.data, where='post', color='r', linestyle='--')
        ax.set_title('minor')

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

    def reset(self):
        self.num_iter = 0
        self.errors = []

    def plot_errors(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.errors, )

    def solve(self):
        self.reset()
        while self.continue_flag == 1:
            self.errors.append(self.update_())
            # self.plot_incremental_comparison()
            # plt.show()
            self.num_iter = self.num_iter + 1
            if self.num_iter == self.num_max_iter or self.error <= self.precision:
                self.continue_flag = 0
                if self.num_iter == self.num_max_iter:
                    self.logger.info('Warning: Maximum iteration reached...Give up...')
                if self.error <= self.precision:
                    self.logger.info('Converge!')

                    self.plot_errors()
                    self.plot_incremental_comparison()
                    self.major_solver.plot_value_surface()
                    self.minor_solver.plot_value_surface()
                    self.major_solver.plot_stop_flag()
                    self.minor_solver.plot_stop_flag()


if __name__ == '__main__':
    root_logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

    for handler in root_logger.handlers:
        handler.setFormatter(fmt)

    bond_model = CCBModel()

    major_stopping_dist = bond_model.get_initial_major_stopping_dist()
    minor_stopping_dist = bond_model.get_initial_minor_stopping_dist()

    solver = GameSolver(bond_model, major_stopping_dist, minor_stopping_dist)
    solver.solve()
    plt.show()
    # solver.plot_equilibrium_strats()
