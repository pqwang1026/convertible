from OptimalStoppingSolver import ConvertibleModel, OptimalStoppingSolver
import numpy as np


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

    def set_stopping_distribution(self, mu):
        self.stopping_solver.model.mu = mu
        self.model.mu = mu

    def get_stopping_distribution(self):
        return self.stopping_solver.stopping_distribution

    def update(self):
        self.last_dist = self.stopping_solver.model.mu
        self.stopping_solver.solve_full()
        self.new_dist = self.get_stopping_distribution()
        self.set_stopping_distribution(self.new_dist)
        self.num_iter = self.num_iter + 1
        self.error = np.linalg.norm(self.last_dist - self.new_dist)
        print self.error
        print self.new_dist
        if self.num_iter == self.max_iter or self.error <= self.precision:
            self.continue_flag = 0
            if self.num_iter == self.max_iter:
                print "Warning: Maximum iteration reached...Give up..."
            if self.error <= self.precision:
                print "Converge!"

    def solve_bond_holder_mfg(self, tau0):
        self.set_call_time(tau0)
        self.set_stopping_distribution(np.linspace(0,1,tau0 + 1))
        self.num_iter = 0
        self.continue_flag = 1
        self.error = 10000.0
        print "Solving bond holder's MFG for call time = ", tau0
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
        print issuer_payment
        return np.argmin(issuer_payment)

    def test_trivial_condition(self):
        return self.model.c < self.model.k * self.model.r / (1 + self.model.r)

    def solve_full(self):
        for tau0 in range(self.model.mat, 1, -1):
            self.solve_bond_holder_mfg(tau0)
            issuer_optimal_call = self.compute_optimal_call_issuer()
            if issuer_optimal_call == tau0:
                self.mfg_solution.update({tau0:self.model.mu})
                print "MFG Equilibrium Found @ Call = ", tau0
            else:
                print "MFG Equilibrium Not Found @ Call = ", tau0
