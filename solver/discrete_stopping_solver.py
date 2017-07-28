import numpy as np
import pandas as pd
import utils.perf as perf
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class DiscreteStoppingModel:
    def __init__(self):
        self.driver = None
        self.terminal_cost = None
        self.running_cost = None


class DiscreteStoppingConfig:
    def __init__(self):
        self.time_num_grids = None
        self.time_lower_bound = None
        self.time_upper_bound = None

        self.state_upper_bound = None
        self.state_lower_bound = None
        self.state_num_grids = None


class DiscreteStoppingSolver:
    def __init__(self, model, config):
        self.driver = model.driver  # driver is a function of (i,x,noise)
        self.running_cost = model.running_cost  # running cost is a function of (i,x)
        self.terminal_cost = model.terminal_cost  # terminal cost is a function of (i,x)

        self.time_num_grids = config.time_num_grids
        self.time_lower_bound = config.time_lower_bound
        self.time_upper_bound = config.time_upper_bound

        self.state_upper_bound = config.state_upper_bound
        self.state_lower_bound = config.state_lower_bound
        self.state_num_grids = config.state_num_grids

        self.value = np.zeros(shape=(self.N + 1, self.M + 1)) * np.nan
        self.stop_flag = np.zeros(shape=(self.N + 1, self.M + 1))

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
        for j in range(0, self.M + 1):
            self.value[self.N][j] = self.terminal_cost(self.time_from_iloc(self.N), self.state_from_iloc(j))
            self.stop_flag[self.N][j] = False

        for i in reversed(range(0, self.N)):
            for j in range(0, self.M + 1):
                t = self.time_from_iloc(i)
                x = self.state_from_iloc(j)
                x_up = self.driver(t, x, 1)
                x_dn = self.driver(t, x, -1)
                value_up = np.interp(x_up, [self.state_from_iloc(k) for k in range(0, self.M + 1)], self.value[i + 1], )
                value_dn = np.interp(x_dn, [self.state_from_iloc(k) for k in range(0, self.M + 1)], self.value[i + 1], )
                value_stop = self.terminal_cost(t, x)
                value_non_stop = (value_up + value_dn) / 2
                self.value[i][j] = max(value_stop, value_non_stop)
                self.stop_flag[i][j] = (value_stop > value_non_stop)

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
                color = 'r'
                marker = 'o'
            ax.plot(t, x, marker=marker, color=color, markersize=4)
        plt.grid()
        plt.title('black for stop, red for non-stop.')
        plt.show()

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

        plt.show()


def put_payoff(t, x, strike, r):
    return np.exp(-r * t) * max(strike - x, 0)


def put_payoff_getter(strike, r):
    def payoff(t, x):
        return put_payoff(t, x, strike, r)

    return payoff


def log_normal_driver(t, x, noise):
    return x + noise


if __name__ == '__main__':
    model = DiscreteStoppingModel()
    config = DiscreteStoppingConfig()

    model.driver = log_normal_driver
    model.terminal_cost = put_payoff_getter(100, 0.001)

    config.time_num_grids = 40
    config.time_upper_bound = 100
    config.time_lower_bound = 0

    config.state_num_grids = 60
    config.state_upper_bound = 200
    config.state_lower_bound = 50

    solver = DiscreteStoppingSolver(model, config)

    solver.solve()
    solver.plot_value_surface()
    solver.plot_stop_flag()
