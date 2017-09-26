import numpy as np
import pandas as pd
import utils.perf as perf
import matplotlib.pyplot as plt
import logging
import utils.distribution

logger = logging.getLogger(__name__)


class StoppingOptimizeType:
    MAXIMIZE = 'MAXIMIZE'
    MINIMIZE = 'MINIMIZE'


class DiscreteStoppingModel:
    """
    This is a Markovian model. We usually start with a continuous model, dX_t = b(t,X_t)dt + sigma(t,X_t)dW_t, and then discretize it by specifying the time_num_grids.
    For an example of defining a new model, c.f. american_option_solver.py.
    A model must implements the following interface:
    -> driver
    -> terminal_reward (optional, default 0)
    -> running_reward (optional, default 0)
    -> update_bounds. Often times we would like to set time/state upper/lower bounds based on some parameters.
    """

    def __init__(self):
        self.time_num_grids = None
        self.time_lower_bound = None
        self.time_upper_bound = None

        self.state_upper_bound = None
        self.state_lower_bound = None
        self.state_num_grids = None

        self.optimize_type = StoppingOptimizeType.MAXIMIZE

    @property
    def time_increment(self):
        return (self.time_upper_bound - self.time_lower_bound) / self.time_num_grids

    @property
    def dt(self):
        return self.time_increment

    @property
    def driver(self):
        """
        This property should return a function of 3 variables: t, x, and noise.
        Assume that all the time grids are defined in this model, then dt is defined (as T / num_grids)
        The driver function dictates that given t, x and a N(0,1) noise, what should be the value at t + dt.
        For example, for a Wiener process with volatility sigma, the driver should be
            lambda t x, noise : x + np.sqrt(self.dt) * noise
        """
        return lambda t, x, noise: 0

    @property
    def terminal_reward(self):
        """
        This property should return a function of 2 variables: t, x.
        It corresponds to the value g(tau, X_tau) in the objective function for us to maximize.
        """
        return lambda t, x: 0

    @property
    def running_reward(self):
        """
        This property shoudl return a function of 2 variables: t, x.
        It corresponds to the value \int^tau_0 f(t, X_t)dt in the objective function for us to maximize.
        """
        return lambda t, x: 0

    def update_bounds(self):
        pass


class DiscreteStoppingSolver:
    def __init__(self, model):
        self.model = model

        self.value = np.zeros(shape=(self.N + 1, self.M + 1)) * np.nan
        self.stop_flag = np.zeros(shape=(self.N + 1, self.M + 1))

        self.stopping_dist = None

    @property
    def optimize_type(self):
        return self.model.optimize_type

    @property
    def driver(self):
        return self.model.driver

    @property
    def terminal_reward(self):
        return self.model.terminal_reward

    @property
    def running_reward(self):
        return self.model.running_reward

    @property
    def time_num_grids(self):
        return self.model.time_num_grids

    @property
    def time_lower_bound(self):
        return self.model.time_lower_bound

    @property
    def time_upper_bound(self):
        return self.model.time_upper_bound

    @property
    def state_num_grids(self):
        return self.model.state_num_grids

    @property
    def state_lower_bound(self):
        return self.model.state_lower_bound

    @property
    def state_upper_bound(self):
        return self.model.state_upper_bound

    @property
    def N(self):
        return self.time_num_grids

    @property
    def state_increment(self):
        return (self.state_upper_bound - self.state_lower_bound) / self.state_num_grids

    @property
    def time_increment(self):
        return (self.time_upper_bound - self.time_lower_bound) / self.time_num_grids

    @property
    def M(self):
        return self.state_num_grids

    def state_from_iloc(self, j):
        return self.state_lower_bound + self.state_increment * j

    def state_to_iloc(self, x):
        j = int(round((x - self.state_lower_bound) / self.state_increment))
        if j > self.state_num_grids:
            j = self.state_num_grids
        if j < 0:
            j = 0
        return j

    def time_to_iloc(self, t):
        j = int(round((t - self.time_lower_bound) / self.time_increment))
        if j > self.time_num_grids:
            j = self.time_num_grids
        if j < 0:
            j = 0
        return j

    def time_from_iloc(self, i):
        return self.time_lower_bound + self.time_increment * i

    def time_state_from_iloc(self, i, j):
        return self.time_from_iloc(i), self.state_from_iloc(j)

    def get_grid_map(self):
        map = dict()
        for i in reversed(range(0, self.N + 1)):
            for j in range(0, self.M + 1):
                map[(i, j)] = self.time_state_from_iloc(i, j)
        return map

    def solve(self):
        self.model.update_bounds()
        self.model.update_bounds()
        logger.info('Start solving, mode = {0}.'.format(self.optimize_type))
        for j in range(0, self.M + 1):
            self.value[self.N][j] = self.model.terminal_reward(self.time_from_iloc(self.N), self.state_from_iloc(j))
            self.stop_flag[self.N][j] = True

        for i in reversed(range(0, self.N)):
            for j in range(0, self.M + 1):
                t = self.time_from_iloc(i)
                x = self.state_from_iloc(j)
                x_up = self.model.driver(t, x, 1)
                x_dn = self.model.driver(t, x, -1)

                value_up = np.interp(x_up, [self.state_from_iloc(k) for k in range(0, self.M + 1)], self.value[i + 1], )
                value_dn = np.interp(x_dn, [self.state_from_iloc(k) for k in range(0, self.M + 1)], self.value[i + 1], )
                value_running = self.model.running_reward(t, x)

                value_stop = self.terminal_reward(t, x)
                value_non_stop = (value_up + value_dn) / 2 + value_running

                if self.optimize_type == StoppingOptimizeType.MAXIMIZE:
                    self.value[i][j] = max(value_stop, value_non_stop)
                    self.stop_flag[i][j] = (value_stop > value_non_stop)
                elif self.optimize_type == StoppingOptimizeType.MINIMIZE:
                    self.value[i][j] = min(value_stop, value_non_stop)
                    self.stop_flag[i][j] = (value_stop < value_non_stop)
                else:
                    raise NotImplementedError

    @perf.timed
    def estimate_stopping_distribution(self, initial_value, num_samples=1000):
        if initial_value > self.state_upper_bound or initial_value < self.state_lower_bound:
            logger.error('Initial value out of bound! initial value = {0}, upper bound = {1}, lower bound = {2}.'.format(initial_value, self.state_upper_bound, self.state_lower_bound))
            raise RuntimeError
        data = []
        for _ in range(num_samples):
            i = 0
            j = self.state_to_iloc(initial_value)
            while not self.stop_flag[i][j]:
                t = self.time_from_iloc(i)
                x = self.state_from_iloc(j)
                noise = np.random.binomial(1, 0.5) * 2 - 1

                x_next = self.model.driver(t, x, noise)

                i += 1
                j = self.state_to_iloc(x_next)
            data.append(self.time_from_iloc(i))
        res = utils.distribution.SampleDistribution(data)
        standard_nodes = [self.time_from_iloc(i) for i in range(0, self.time_num_grids + 1)]
        standard_prob = [res.pdf.get(node, 0) for node in standard_nodes]
        standard_res = utils.distribution.Distribution(standard_nodes, standard_prob)
        return standard_res

    def plot_stop_flag(self):
        grid_map = self.get_grid_map()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for (i, j), (t, x) in grid_map.items():
            stop_flag = self.stop_flag[i][j]
            if stop_flag:
                color = 'k'
                marker = 'D'
            else:
                color = 'c'
                marker = 'o'
            ax.plot(t, x, marker=marker, color=color, markersize=4)
        plt.grid()
        plt.title(type(self.model).__name__ + '(black for stop, green for non-stop.)')

    def plot_value_surface(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X_ = np.array([self.time_from_iloc(i) for i in range(0, self.N + 1)])
        Y_ = np.array([self.state_from_iloc(j) for j in range(0, self.M + 1)])
        X, Y = np.meshgrid(X_, Y_)
        X = X.transpose()
        Y = Y.transpose()
        Z = self.value

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.suptitle(type(self.model).__name__)


def put_payoff(t, x, strike, r):
    return np.exp(-r * t) * max(strike - x, 0)


def put_payoff_getter(strike, r):
    def payoff(t, x):
        return put_payoff(t, x, strike, r)

    return payoff


def log_normal_driver(t, x, noise):
    return x + noise


if __name__ == '__main__':
    pass
