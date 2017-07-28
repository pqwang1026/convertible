import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class ConvertibleModel(object):
    def __init__(self, v0, tau0, r, nu, c, k, sigma, delta, mu, mat):
        self.v0 = v0  # initial capital
        self.mat = mat  # bond's maturity
        self.tau0 = tau0  # firm's call time
        self.r = r  # discount rate
        self.nu = nu  # firm's capital growth rate
        self.c = c  # coupon rate
        self.k = k  # penalty rate for early call
        self.sigma = sigma  # precision of valuation
        self.delta = delta  # shares per unit of bond after conversion, i.e. conversion ratio
        self.mu = mu  # mean field: how many bonds have been converted


class OptimalStoppingSolver(object):
    def __init__(self, model, grid_size, monte_carlo):
        self.model = model  # model of convertible bond
        self.grid_size = grid_size  # number of discretization of space grid
        self.monte_carlo = monte_carlo  # number of monte carlo to compute the distribution of optimal stopping

        # set up the grid and the solution array
        self.grid_bound = np.ceil(model.v0 * np.power((1.0 + model.nu + model.sigma), model.mat))
        self.grid_disc = self.grid_bound / self.grid_size
        self.grid = np.linspace(start=0.0, stop=self.grid_bound, num=self.grid_size + 1)
        self.value_function = np.zeros(shape=(model.mat + 1, self.grid_size + 1))
        self.optimal_strat = np.zeros(shape=(model.mat + 1, self.grid_size + 1), dtype=int)
        self.conversion_boundary = np.zeros(shape=model.mat + 1)
        self.stopping_distribution = np.zeros(shape=model.mat + 1)
        self.process_cost = np.array([0])
        self.process_liab = np.array([0])
        self.call_payoff = 0

    def prepare_model(self):
        if self.model.tau0 == self.model.mat:
            self.call_payoff = 1.0
        else:
            self.call_payoff = self.model.k

        # set processes l and c
        self.process_cost = np.zeros(shape=(self.model.tau0 + 1))
        self.process_liab = np.zeros(shape=(self.model.tau0 + 1))
        for t in range(self.model.tau0):
            self.process_cost[t] = self.model.c * (1.0 - self.model.mu[t])
            self.process_liab[t] = 1.0 - self.model.mu[t]
        self.process_cost[self.model.tau0] = (1.0 - self.model.mu[self.model.tau0]) * self.call_payoff

        # set terminal condition
        for n in range(1 + self.grid_size):
            conversion_payoff = self.get_conversion_payoff(self.model.tau0, n)
            self.value_function[self.model.tau0, n] = max(conversion_payoff, self.call_payoff)
            if conversion_payoff >= self.call_payoff:
                self.optimal_strat[self.model.tau0, n] = 1

    def get_conversion_payoff(self, t, n):
        return self.model.delta * (1.0 - self.model.delta * self.model.mu[t]) * (self.grid[n] - self.process_liab[t])

    # linear interpolation
    def get_value_function(self, t, x):
        return np.interp(x, self.grid, self.value_function[t,])

    # compute the value function at time t
    def update(self, t, n):
        x_up = self.grid[n] * (1.0 + self.model.nu + self.model.sigma) - self.process_cost[t + 1]
        x_down = x_up - 2.0 * self.grid[n] * self.model.sigma
        go_on_payoff = self.model.c + 0.5 * (self.get_value_function(t + 1, x_up) + self.get_value_function(t + 1, x_down)) / (1.0 + self.model.r)
        conversion_payoff = self.get_conversion_payoff(t, n)
        # print "Continue payoff = ", go_on_payoff, "Conversion payoff = ", conversion_payoff
        self.value_function[t, n] = max(conversion_payoff, go_on_payoff)
        if conversion_payoff >= go_on_payoff:
            self.optimal_strat[t, n] = 1

    def solve_optimal_stopping(self):
        for t in range(self.model.tau0 - 1, -1, -1):
            for n in range(self.grid_size + 1):
                self.update(t, n)

    def get_conversion_boundary(self):
        for t in range(self.model.tau0, -1, -1):
            try:
                self.conversion_boundary[t] = np.ndarray.min(np.where(self.optimal_strat[t,] == 1)[0])
            except ValueError:
                self.conversion_boundary[t] = self.grid_size * 2

    def get_stopping_flag(self, x, t):
        left_index = np.int(np.floor(x / self.grid_disc))
        return self.optimal_strat[t, left_index] * self.optimal_strat[t, left_index + 1]

    def estimate_stopping_distribution(self):
        self.stopping_distribution = np.zeros(shape=self.model.tau0 + 1)
        for k in range(self.monte_carlo):
            t = 0
            v = self.model.v0
            while self.get_stopping_flag(v, t) == 0 and t < self.model.tau0:
                t = t + 1
                v = v * (1.0 + self.model.nu + self.model.sigma * np.random.binomial(1, 0.5)) - self.process_cost[t]
            if t < self.model.tau0:
                self.stopping_distribution[t] = self.stopping_distribution[t] + 1.0
            else:
                if self.get_stopping_flag(v, t) == 1:
                    self.stopping_distribution[t] = self.stopping_distribution[t] + 1.0
        self.stopping_distribution = np.cumsum(self.stopping_distribution / self.monte_carlo)

    def plot_conversion_boundary(self):
        time_line = np.linspace(0, self.model.tau0, self.model.tau0 + 1)
        plt.plot(time_line, self.conversion_boundary * self.grid_disc)

    def plot_value_function(self, t):
        conversion_payoff = np.zeros(shape=self.grid_size + 1)
        for n in range(self.grid_size + 1):
            conversion_payoff[n] = self.get_conversion_payoff(t, n)
        plt.plot(self.grid, self.value_function[t,])
        plt.plot(self.grid, conversion_payoff)

    def plot_stopping_distribution(self):
        time_line = np.linspace(0, self.model.tau0, self.model.tau0 + 1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_line, self.stopping_distribution)
        plt.show()

    # wrapper function of solution
    def solve_full(self):
        self.prepare_model()
        self.solve_optimal_stopping()
        self.get_conversion_boundary()
        self.estimate_stopping_distribution()
