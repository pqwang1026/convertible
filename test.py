import numpy as np
import matplotlib.pyplot as plt
import logging

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
    print(handler)
    handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StoppingModel(object):
    def __init__(self, horizon, initial_condition):
        self.horizon = horizon
        self.initial_condition = initial_condition

    def dynamic(self, t, x, noise):
        raise NotImplementedError

    def running_payoff(self, t, x):
        raise NotImplementedError

    def terminal_payoff(self, t, x):
        raise NotImplementedError


class ConvertibleModel(object):
    def __init__(self, v0, tau0, r, R, nu, c, dividend, k, sigma, delta, mu, mat, R0, sigma0):
        self.v0 = v0        # initial capital
        self.mat = mat      # bond's maturity
        self.tau0 = tau0    # firm's call time
        self.r = r          # discount rate
        self.nu = nu        # firm's capital growth rate
        self.c = c          # coupon rate
        self.R0 = R0        # initial refinance interest rate
        self.k = k          # penalty rate for early call
        self.sigma = sigma  # precision of valuation
        self.sigma0 = sigma0 # volatility of interest rate
        self.delta = delta  # shares per unit of bond after conversion
        self.dividend = dividend # dividend per share
        self.mu = mu        # mean field: how many bonds have been converted


class MinorStoppingModel(StoppingModel):
    def __abs__(self, convertible_model, major_stopping_dist, minor_stopping_dist):
        StoppingModel.__init__(convertible_model.mat + 1, convertible_model.v0)
        self.major_stopping_dist = major_stopping_dist
        self.minor_stopping_dist = minor_stopping_dist

    def dynamic(self, t, x, noise):
        return 0

    def running_payoff(self, t, x):
        return 0

    def terminal_payoff(self, t, x):
        return 0


class MajorStoppingModel(StoppingModel):
    def __abs__(self, convertible_model, major_stopping_dist, minor_stopping_dist):
        StoppingModel.__init__(convertible_model.mat, convertible_model.R0)
        self.major_stopping_dist = major_stopping_dist
        self.minor_stopping_dist = minor_stopping_dist

    def dynamic(self, t, x, noise):
        return 0

    def running_payoff(self, t, x):
        return 0

    def terminal_payoff(self, t, x):
        return 0


class OptimalStoppingSolver(object):
    def __init__(self, stopping_model, grid_size, monte_carlo):
        self.stopping_model = stopping_model  # model of convertible bond
        self.num_grids = grid_size  # number of discretization of space grid
        self.monte_carlo = monte_carlo  # number of monte carlo to compute the distribution of optimal stopping

        # set up the grid and the solution array
        self.grid_bound = np.ceil(self.stopping_model.v0 * np.power((1.0 + self.stopping_model.nu + self.stopping_model.sigma), self.stopping_model.mat))
        self.grid_size = self.grid_bound / self.num_grids
        self.grid = np.linspace(start=0.0, stop=self.grid_bound, num=self.num_grids + 1)
        self.value_function = np.zeros(shape=(stopping_model.horizon + 1, self.num_grids + 1))
        self.optimal_strat = np.zeros(shape=(stopping_model.horizon + 1, self.num_grids + 1), dtype=int)
        self.conversion_boundary = np.zeros(shape=stopping_model.horizon + 1)
        self.stopping_distribution = np.zeros(shape=stopping_model.horizon + 1)

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
        for n in range(1 + self.num_grids):
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
        x_up = self.stopping_model.dynamic(t, self.grid[n], 1)
        x_down = self.stopping_model.dynamic(t, self.grid[n], -1)
        continue_payoff = self.stopping_model.running_payoff(t, self.grid[n]) + \
                          0.5 * (self.get_value_function(t + 1, x_up) + self.get_value_function(t + 1, x_down)) / (1.0 + self.model.r)
        stop_payoff = self.stopping_model.terminal_payoff(t, self.grid[n])
        # print "Continue payoff = ", go_on_payoff, "Conversion payoff = ", conversion_payoff
        self.value_function[t, n] = max(continue_payoff, stop_payoff)
        if stop_payoff >= continue_payoff:
            self.optimal_strat[t, n] = 1

    def solve_optimal_stopping(self):
        for n in range(self.num_grids + 1):
            self.value_function[self.stopping_model.horizon, n] = self.stopping_model.running_payoff(self.stopping_model.horizon, self.grid[n])
            self.optimal_strat[self.stopping_model.horizon, n] = 1
        for t in range(self.stopping_model.horizon - 1, -1, -1):
            for n in range(self.num_grids + 1):
                self.update(t, n)

    def get_conversion_boundary(self):
        for t in range(self.stopping_model.horizon, -1, -1):
            try:
                self.conversion_boundary[t] = np.ndarray.min(np.where(self.optimal_strat[t,] == 1)[0])
            except ValueError:
                self.conversion_boundary[t] = self.num_grids * 2

    def get_stopping_flag(self, x, t):
        left_index = np.int(np.floor(x / self.grid_size))
        return self.optimal_strat[t, left_index] * self.optimal_strat[t, left_index + 1]

    def estimate_stopping_distribution(self):
        self.stopping_distribution = np.zeros(shape=self.stopping_model.horizon + 1)
        for k in range(self.monte_carlo):
            t = 0
            v = self.stopping_model.initial_condition
            while self.get_stopping_flag(v, t) == 0 and t < self.model.tau0:
                t = t + 1
                v = v * (1.0 + self.model.nu + self.model.sigma * np.random.binomial(1, 0.5)) - self.process_cost[t]
            self.stopping_distribution[t] = self.stopping_distribution[t] + 1.0
        self.stopping_distribution = np.cumsum(self.stopping_distribution / self.monte_carlo)

    def plot_conversion_boundary(self):
        time_line = np.linspace(0, self.model.tau0, self.model.tau0 + 1)
        plt.plot(time_line, self.conversion_boundary * self.grid_size)

    def plot_value_function(self, t):
        stop_payoff = np.zeros(shape=self.num_grids + 1)
        for n in range(self.num_grids + 1):
            stop_payoff[n] = self.stopping_model.terminal_payoff(t, self.grid[n])
        plt.plot(self.grid, self.value_function[t,])
        plt.plot(self.grid, stop_payoff)

    def plot_stopping_distribution(self):
        time_line = np.linspace(0, self.model.tau0, self.model.tau0 + 1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_line, self.stopping_distribution)
        plt.show()

    def solve_full(self):
        """
        Public interface
        """
        self.prepare_model()
        self.solve_optimal_stopping()
        self.get_conversion_boundary()
        self.estimate_stopping_distribution()
