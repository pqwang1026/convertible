import numpy as np
import logging
from test import MinorStoppingModel, MajorStoppingModel, OptimalStoppingSolver

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
    print(handler)
    handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MFSGSolver:
    def __init__(self, T):
        self.minor_model = MinorStoppingModel()
        self.major_model = MajorStoppingModel()

        self.minor_stopping_dist = None
        self.major_stopping_dist = None

        self.minor_stopping_dist_prev = None
        self.major_stopping_dist_prev = None

        self.num_mc = 1e3  # number of monte carlo trials

        self.T = T

    def update(self):
        self.minor_stopping_dist_prev = self.minor_stopping_dist
        self.major_stopping_dist_prev = self.major_stopping_dist

        self.minor_model = MinorStoppingModel(self.T, self.major_stopping_dist, self.minor_stopping_dist)
        self.major_model = MajorStoppingModel(self.T, self.minor_stopping_dist)

        major_solver = OptimalStoppingSolver(self.major_model, 1, self.num_mc)
        minor_solver = OptimalStoppingSolver(self.minor_model, 1, self.num_mc)

        major_solver.solve_full()
        minor_solver.solve_full()

        self.minor_stopping_dist = minor_solver.stopping_distribution
        self.major_stopping_dist = major_solver.stopping_distribution

    def plot_comparison_

    @property
    def logger(self):
        return logger


class MeanFieldGameSolver(object):
    def __init__(self, model, grid_size, monte_carlo, precision, max_iter):
        self.model = model
        self.stopping_solver = OptimalStoppingSolver(model, grid_size, monte_carlo)
        self.precision = precision
        self.max_iter = max_iter
        self.num_iter = 0
        self.continue_flag = 1
        self.error = 10000.0
        self.last_dist = np.array([0])
        self.new_dist = np.array([0])
        self.mfg_solution = {}

    def set_call_time(self, tau0):
        self.stopping_solver.model.tau0 = tau0
        self.model.tau0 = tau0
        self.logger.info('Set model tau0 to {0}'.format(tau0))

    def set_stopping_distribution(self, mu):
        self.stopping_solver.model.mu = mu
        self.model.mu = mu
        self.logger.info('Set model mu to {0}'.format(mu))

    def get_stopping_distribution(self):
        return self.stopping_solver.stopping_distribution

    def update(self):
        self.last_dist = self.stopping_solver.model.mu
        self.stopping_solver.solve_full()
        self.new_dist = self.get_stopping_distribution()
        self.set_stopping_distribution(self.new_dist)
        self.num_iter = self.num_iter + 1
        self.error = np.linalg.norm(self.last_dist - self.new_dist)
        self.logger.info(self.error)
        self.logger.info(self.new_dist)
        if self.num_iter == self.max_iter or self.error <= self.precision:
            self.continue_flag = 0
            if self.num_iter == self.max_iter:
                self.logger.info('Warning: Maximum iteration reached...Give up...')
            if self.error <= self.precision:
                self.logger.info('Converge!')

    def solve_bond_holder_mfg(self, tau0):
        self.set_call_time(tau0)
        self.set_stopping_distribution(np.linspace(0, 1, tau0 + 1))
        self.num_iter = 0
        self.continue_flag = 1
        self.error = 10000.0
        self.logger.info('Solving bond holder MFG for call time = {0}'.format(tau0))
        while self.continue_flag == 1:
            self.update()

    def compute_optimal_call_issuer(self):
        principle_payment = np.ones(self.model.mat + 1)
        principle_payment[self.model.mat] = self.model.k
        mu_extend = np.zeros(self.model.mat + 1)
        mu_extend[0:self.model.tau0 + 1] = self.model.mu
        if self.model.tau0 < self.model.mat:
            mu_extend[self.model.tau0 + 1: self.model.mat + 1] = mu_extend[self.model.tau0] * np.ones(self.model.mat - self.model.tau0)
        mu_extend = np.round(mu_extend, 8)
        discounting = np.power((1.0 + self.model.r), -np.linspace(0, self.model.mat, self.model.mat + 1))
        cum_coupon_payment = np.cumsum(self.model.c * discounting * (1.0 - mu_extend))
        issuer_payment = cum_coupon_payment + principle_payment * discounting * (1.0 - mu_extend)
        self.logger.info('Issuer payment on call date = {0}'.format(issuer_payment))
        return np.argmin(issuer_payment)

    def test_trivial_condition(self):
        return self.model.c < self.model.k * self.model.r / (1 + self.model.r)

    def solve_full(self):
        for tau0 in range(self.model.mat, 1, -1):
            self.solve_bond_holder_mfg(tau0)
            issuer_optimal_call = self.compute_optimal_call_issuer()
            if issuer_optimal_call == tau0:
                self.mfg_solution.update({tau0: self.model.mu})
                self.logger.info('MFG Equilibrium Found @ Call = {0}'.format(tau0))
            else:
                self.logger.info('MFG Equilibrium Not Found @ Call = {0}'.format(tau0))

    @property
    def logger(self):
        return logger
