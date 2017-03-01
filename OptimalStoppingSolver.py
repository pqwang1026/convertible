import numpy as np
import matplotlib.pyplot as plt


class ConvertibleModel(object):
    def __init__(self, l0, v0, tau0, r, nu, c, d, k, lbd, sigma, delta, mu, mat):
        self.l0 = l0        # initial debt
        self.v0 = v0        # initial capital
        self.mat = mat      # bond's maturity
        self.tau0 = tau0    # firm's call time
        self.r = r          # discount rate
        self.nu = nu        # firm's capital growth rate
        self.c = c          # coupon rate
        self.d = d          # debt's unit nominal
        self.k = k          # penalty rate for early call
        self.lbd = lbd      # dilution coef
        self.sigma = sigma  # precision of valuation
        self.delta = delta  # shares per unit of bond after conversion
        self.mu = mu        # mean field: how many bonds have been converted


class OptimalStoppingSolver(object):
    def __init__(self, model, grid_size, monte_carlo):
        self.model = model                  # model of convertible bond
        self.grid_size = grid_size          # number of discretization of space grid
        self.monte_carlo = monte_carlo       # number of monte carlo to compute the distribution of optimal stopping

        # set up the grid and the solution array
        self.grid_bound = model.v0 * np.power(model.nu + model.sigma, model.tau0)
        self.grid_disc = self.grid_bound / self.grid_size
        self.grid = np.linspace(start = 0, stop = self.grid_bound, num = self.grid_size)
        self.value_function = np.zeros(shape = (model.tau0 + 1, self.grid_size + 1))
        self.conversion_boundary = np.zeros(shape = model.tau0 + 1)
        self.stopping_distribution = np.zeros(shape = model.tau0 + 1)

        if model.tau0 == model.mat:
            self.call_payoff = 1.0
        else:
            self.call_payoff = model.k

        # set processes l and c
        self.process_cost = np.zeros(shape = (model.tau0 + 1))
        self.process_liab = np.zeros(shape = (model.tau0 + 1))
        for t in range(model.tau0):
            self.process_cost[t] = model.c * model.l0 * (1.0 - model.mu[t])
            self.process_liab[t] = model.l0 * (1.0 - model.mu[t])
        self.process_cost[model.tau0] = model.l0 * (1.0 - model.mu[model.tau0]) * self.call_payoff

        # set terminal condition
        for n in range(1 + self.grid_size):
            self.value_function[model.tau, n] = max(self.get_conversion_payoff(model.tau, n), model.d * self.call_payoff)

    def get_conversion_payoff(self, t, n):
        return self.model.delta * (1.0 - self.model.lbd * self.model.mu[t]) * (self.grid[n] - self.process_liab[t])

    # linear interpolation
    def get_value_function(self, t, x):
        return np.interp(x, self.grid, self.value_function[t,])

    # compute the value function at time t
    def update(self, t, n):
        x_up = self.grid[n] * (1.0 + self.model.nu + self.model.sigma) - self.process_cost[t + 1]
        x_down = x_up - 2.0 * self.grid[n] * self.model.sigma
        contin_payoff = self.model.c * self.model.d + \
                        0.5 * (self.get_value_function(t + 1, x_up) + self.get_value_function(t + 1, x_down)) / (1 + self.model.r)
        return max(self.get_conversion_payoff(t, n), contin_payoff)

    # wrapper function for solution
    def solve(self):
        for t in range(self.model.tau0 - 1, -1, -1):
            for n in range(self.grid_size + 1):
                self.update(t, n)

    def get_conversion_boundary(self):
        for t in range(self.model.tau0, -1, -1):
            n = 0
            while self.get_conversion_payoff(t, n) < self.value_function[t, n]:
                n = n + 1
            self.conversion_boundary[t] = n

    def plot_conversion_boundary(self):
        time_line = np.linspace(0, self.model.tau0, self.model.tau0 + 1)
        plt.plot(time_line, self.conversion_boundary * self.grid_disc)

    def estimate_stopping_distribution(self):
        for k in range(self.monte_carlo):
            t = 0
            v = self.model.v0
            boundary = self.conversion_boundary[t] * self.grid_disc
            while v < boundary and t < self.model.tau0:
                t = t + 1
                boundary = self.conversion_boundary[t] * self.grid_disc
                v = v * (1.0 + self.model.nu + self.model.sigma * np.random.binomial(1, 0.5)) - self.process_cost[t]
            if t < self.model.tau0:
                self.stopping_distribution[t] = self.stopping_distribution[t] + 1.0
            else:
                if v >= boundary:
                    self.stopping_distribution[t] = self.stopping_distribution[t] + 1.0
        self.stopping_distribution = self.stopping_distribution / self.monte_carlo










