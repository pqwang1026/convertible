import numpy as np
import matplotlib.pyplot as plt
import logging
import distribution as dist

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
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
    def __init__(self, v0, r, nu, c, dividend, sigma, delta, mat, r0, sigma0):
        self.v0 = v0  # initial capital
        self.mat = mat  # bond's maturity
        self.r = r  # discount rate
        self.nu = nu  # firm's capital growth rate
        self.c = c  # coupon rate
        self.R0 = r0  # initial refinance interest rate
        self.sigma = sigma  # precision of valuation
        self.sigma0 = sigma0  # volatility of interest rate
        self.delta = delta  # shares per unit of bond after conversion
        self.dividend = dividend  # dividend per share


class MinorStoppingModel(StoppingModel):
    def __init__(self, convertible_model, major_stopping_dist, minor_stopping_dist):
        super().__init__(convertible_model.mat + 1, convertible_model.v0)
        self.major_stopping_dist = major_stopping_dist
        self.minor_stopping_dist = minor_stopping_dist
        self.convertible_model = convertible_model

    def dynamic(self, t, x, noise):
        return (1.0 + self.convertible_model.nu + self.convertible_model.sigma * noise) * x - \
               self.convertible_model.c * (1 - self.minor_stopping_dist.cdf[t])

    def running_payoff(self, t, x):
        if t == 0:
            return np.power((1.0 + self.convertible_model.r), -t) * self.convertible_model.c
        else:
            return np.power((1.0 + self.convertible_model.r), -t) * self.convertible_model.c * (1.0 - self.major_stopping_dist.cdf[t - 1])

    def terminal_payoff(self, t, x):
        if t == 0:
            return self.convertible_model.delta * (1.0 - self.convertible_model.delta * self.minor_stopping_dist.cdf[0]) * \
                   (x - (1.0 - self.minor_stopping_dist.cdf[0]))
        elif t == self.horizon:
            return np.sum(self.major_stopping_dist.pdf * np.power((1.0 + self.convertible_model.r), -np.linspace(0, self.horizon - 1, self.horizon)))

        else:
            return np.sum(self.major_stopping_dist.pdf[0:t] * np.power((1.0 + self.convertible_model.r), -np.linspace(0, t - 1, t))) + \
                   self.convertible_model.delta * (1.0 - self.convertible_model.delta * self.minor_stopping_dist.cdf[t]) * \
                   (x - (1.0 - self.minor_stopping_dist.cdf[t])) * (1.0 - self.major_stopping_dist.cdf[t - 1])


class MajorStoppingModel(StoppingModel):
    def __init__(self, convertible_model, minor_stopping_dist):
        super().__init__(convertible_model.mat, convertible_model.R0)
        self.minor_stopping_dist = minor_stopping_dist
        self.convertible_model = convertible_model

    def dynamic(self, t, x, noise):
        return x + self.convertible_model.sigma0 * noise

    def running_payoff_raw(self, t, x):
        return np.power((1.0 + self.convertible_model.r), -t) * \
               (self.convertible_model.c + (self.convertible_model.dividend - self.convertible_model.c) * self.minor_stopping_dist.cdf[t])

    def terminal_payoff_raw(self, t, x):
        return (np.power((1.0 + self.convertible_model.r), -t) - np.power((1.0 + self.convertible_model.r), -self.horizon)) * \
               (x + (self.convertible_model.dividend * self.convertible_model.delta - x) * self.minor_stopping_dist.cdf[t]) / self.convertible_model.r \
               + self.running_payoff(t, x)

    def running_payoff(self, t, x):
        return -self.running_payoff_raw(t, x)

    def terminal_payoff(self, t, x):
        return -self.terminal_payoff_raw(t, x)


class OptimalStoppingSolver(object):
    def __init__(self, stopping_model, grid_upper_bound, grid_lower_bound, grid_num, monte_carlo):
        self.stopping_model = stopping_model  # model of convertible bond
        self.num_grids = grid_num  # number of discretization of space grid
        self.monte_carlo = monte_carlo  # number of monte carlo to compute the distribution of optimal stopping

        # set up the grid and the solution array
        self.grid_upper_bound = grid_upper_bound
        self.grid_lower_bound = grid_lower_bound
        self.grid_size = (self.grid_upper_bound - self.grid_lower_bound) / self.num_grids
        self.grid = np.linspace(start=grid_upper_bound, stop=grid_lower_bound, num=self.num_grids + 1)
        self.value_function = np.zeros(shape=(stopping_model.horizon + 1, self.num_grids + 1))
        self.optimal_strat = np.zeros(shape=(stopping_model.horizon + 1, self.num_grids + 1), dtype=int)
        self.conversion_boundary = np.zeros(shape=stopping_model.horizon + 1)
        self.stopping_distribution = np.zeros(shape=stopping_model.horizon + 1)

    def plot_stop_region(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for t in range(0, self.stopping_model.horizon + 1):
            for j in range(0, self.num_grids + 1):
                if self.optimal_strat[t, j] == 1:
                    color = 'k'
                    marker = 'D'
                else:
                    color = 'r'
                    marker = 'o'
                ax.plot([t], [self.grid[j]], color=color, marker=marker)
        plt.show()

    # linear interpolation
    def get_value_function(self, t, x):
        return np.interp(x, self.grid, self.value_function[t,])

    # compute the value function at time t
    def update(self, t, n):
        x_up = self.stopping_model.dynamic(t, self.grid[n], 1.0)
        x_down = self.stopping_model.dynamic(t, self.grid[n], -1.0)
        continue_payoff = self.stopping_model.running_payoff(t, self.grid[n]) + \
                          0.5 * (self.get_value_function(t + 1, x_up) + self.get_value_function(t + 1, x_down))
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
        left_index = np.int(np.floor((x - self.grid_lower_bound) / self.grid_size))
        return self.optimal_strat[t, left_index] * self.optimal_strat[t, left_index + 1]

    def estimate_stopping_distribution(self):
        dist_temp = np.zeros(self.stopping_model.horizon + 1)
        for k in range(self.monte_carlo):
            t = 0
            v = self.stopping_model.initial_condition
            while self.get_stopping_flag(v, t) == 0:
                t = t + 1
                noise = np.random.binomial(1, 0.5) * 2 - 1
                v = self.stopping_model.dynamic(t, v, noise)
            dist_temp[t] = dist_temp[t] + 1.0
        self.stopping_distribution = dist.DiscreteDistribution(dist_temp / self.monte_carlo)

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
        self.stopping_distribution.plot_cdf()

    def solve_full(self):
        """
        Public interface
        """
        self.solve_optimal_stopping()
        self.get_conversion_boundary()
        self.estimate_stopping_distribution()


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
    minor_model = MinorStoppingModel(bond_model, major_stopping_dist, minor_stopping_dist)
    major_model = MajorStoppingModel(bond_model, minor_stopping_dist)

    upper_bound = 20
    lower_bound = 0
    num_grids = 100
    num_mc = 1000
    minor_stopping_solver = OptimalStoppingSolver(minor_model, upper_bound, lower_bound, num_grids, num_mc)
    major_stopping_solver = OptimalStoppingSolver(major_model, upper_bound, lower_bound, num_grids, num_mc)

    major_stopping_solver.solve_full()
    major_stopping_solver.plot_stopping_distribution()
    major_stopping_solver.plot_stop_region()
