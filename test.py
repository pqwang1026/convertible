import utils.stopping_time as st
import seaborn as sns
import numpy as np
import utils.distribution
import matplotlib.pyplot as plt
from collections import Counter

sns.set_style('whitegrid')


class SimpleStoppingSolver:
    def __init__(self, num_steps):
        self.n = num_steps
        self.terminal_payoff = lambda t, x: 0

        self.value = np.zeros(shape=(self.n + 1, self.n + 1)) * np.nan
        self.stop_flag = np.zeros(shape=(self.n + 1, self.n + 1)) * np.nan

    def iloc_to_loc(self, i, j):
        return (i, -i + 2 * j)

    def get_grid_map(self):
        map = dict()
        for i in reversed(range(0, self.n + 1)):
            for j in range(0, i + 1):
                map[(i, j)] = self.iloc_to_loc(i, j)
        return map

    def solve(self):
        for j in range(0, self.n + 1):
            self.value[self.n][j] = self.terminal_payoff(self.n, self.iloc_to_loc(self.n, j)[1])
            self.stop_flag[self.n][j] = True
        for i in reversed(range(0, self.n)):
            for j in range(0, i + 1):
                continue_value = 0.5 * (self.value[i + 1][j + 1] + self.value[i + 1][j])
                stop_value = self.terminal_payoff(i, self.iloc_to_loc(i, j)[1])

                if stop_value > continue_value:
                    self.stop_flag[i][j] = True
                else:
                    self.stop_flag[i][j] = False

                self.value[i][j] = max(continue_value, stop_value)

    def get_st_from_path(self, path: st.Path):
        iloc = (0, 0)
        for i in range(0, self.n):
            if self.stop_flag[iloc[0]][iloc[1]]:
                return self.iloc_to_loc(iloc[0], iloc[1])
            if path[i] == 1:
                iloc = (iloc[0] + 1, iloc[1] + 1)
            else:
                iloc = (iloc[0] + 1, iloc[1])
        return self.iloc_to_loc(iloc[0], iloc[1])

    def get_st(self):
        all_paths = st.get_all_paths(self.n)
        sts = [self.get_st_from_path(path)[0] for path in all_paths]
        return st.StoppingTime(dict(zip(all_paths, sts)))

    def plot_stop_flag(self):
        grid_map = self.get_grid_map()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for (i, j), (t, x) in grid_map.items():
            stop_flag = self.stop_flag[i][j]
            if stop_flag == True:
                color = 'k'
                marker = 'D'
            else:
                color = 'c'
                marker = 'o'
            ax.plot(t, x, marker=marker, color=color, markersize=4)
        plt.title('(black for stop, green for non-stop.)')

    def get_stopping_dist(self):
        all_paths = st.get_all_paths(self.n)
        sts = [self.get_st_from_path(path)[0] for path in all_paths]

        ct = Counter(sts)

        dist = utils.distribution.Distribution(list(ct.keys()), np.array(list(ct.values())) / 2 ** self.n)
        return dist


def get_stopping_dist_from_st(stopping_time: st.StoppingTime):
    times = stopping_time.values()
    ct = Counter(times)
    dist = utils.distribution.Distribution(list(ct.keys()), np.array(list(ct.values())) / 2 ** len(list(stopping_time.keys())[0]))
    return dist


all_st = st.get_all_stopping_times(1)
empty_profile = [0 for _ in range(0, 2 ** 1)]
profile = [0, 1]

I = utils.distribution.Distribution([0, 1], [0.8, 0.2])
rate = 1
while True:
    solver = SimpleStoppingSolver(1)

    solver.terminal_payoff = lambda t, x: 1 / 3 * x * x - I(t)

    solver.solve()
    optimal_st = solver.get_st()

    target_profile = [0 for _ in range(0, 2 ** 1)]
    target_profile[all_st.index(optimal_st)] = 1

    profile = (1 - rate) * np.array(profile) + rate * np.array(target_profile)

    I = utils.distribution.Distribution([0], [0])
    for p, stopping_time in zip(profile, all_st):
        d = get_stopping_dist_from_st(stopping_time)
        I = I + d.mult(p)
    I.plot_cdf()
    print(profile)

    # solver.plot_stop_flag()
    # I.plot_cdf()
    # plt.show()
