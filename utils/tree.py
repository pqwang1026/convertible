import numpy as np
import seaborn as sns
import logging
import scipy
import pandas as pd
import matplotlib.pyplot as plt

root_logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
fmt = logging.Formatter('{asctime} [{levelname}] <{name}> {message}', style='{')

for handler in root_logger.handlers:
    handler.setFormatter(fmt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RandomWalkTree:
    """
    This is a standard additive random walk tree.
    """

    def __init__(self, T, N, increment):
        self.T = T
        self.N = N
        self.p_u = 0.5
        self.p_d = 0.5
        self.increment = increment
        self.dt = T / N

    @property
    def logger(self):
        return logger

    def coordinate_from_iloc(self, i, j, init_value):
        """
        This function returns the (x,y) coordinate of the tree, given (i,j) which is the index in the tree.
        i can be interpreted as the i-th step, and j can be interpreted as the number of up steps.
        """
        return self.dt * i, init_value + (2 * j - i) * self.increment

    def generate_one_path(self, init_value):
        x = init_value
        index = [0]
        data = [x]

        bernoulli = scipy.stats.bernoulli(self.p_u)

        for i in range(1, self.N + 1):
            rv = bernoulli.rvs(1)[0]
            if rv:
                x += self.increment
            else:
                x -= self.increment
            index.append(i * self.dt)
            data.append(x)

        res = pd.Series(data=data, index=index)
        return res

    def plot_tree(self, init_value):
        grid_map = self.get_grid_map(init_value=init_value)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for iloc, coordinate in grid_map.items():
            stop_flag = self.stop_flags[iloc]
            if stop_flag:
                color = 'k'
                marker = 'D'
            else:
                color = 'r'
                marker = 'o'
            ax.plot([coordinate[0]], [coordinate[1]], marker=marker, color=color, markersize=4)
        plt.show()


class MarkovianTree:
    def __init__(self, T, N, increment, f):
        """
        f is the driver function, i.e. X_t = f(t, X_{t-1}, eps_t)
        """
        self.T = T
        self.N = N
        self.p_u = 0.5
        self.p_d = 0.5
        self.increment = increment
        self.dt = T / N
        self.f = f

    def generate_one_path(self, init_value):
        x = init_value
        index = [0]
        data = [x]

        bernoulli = scipy.stats.bernoulli(self.p_u)

        for i in range(1, self.N + 1):
            rv = bernoulli.rvs(1)[0]
            if rv:
                eps = self.increment
            else:
                eps = -self.increment
            index.append(i * self.dt)
            data.append(self.f(i * self.dt, data[-1], eps))

        res = pd.Series(data=data, index=index)
        return res


if __name__ == '__main__':
    def bs_driver(t, x, eps):
        return x * (1 + eps)


    # tree = RandomWalkTree(10, 100, 1)
    tree = MarkovianTree(1, 1000, 0.002, bs_driver)
    path = tree.generate_one_path(1)
    plt.plot(path.index, path.values)
    plt.show()
