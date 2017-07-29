from solver.discrete_stopping_solver import *
import pprint
import utils.perf as perf
import logging

logger = logging.getLogger(__name__)

stream_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s> %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.DEBUG)

handlers = [stream_handler]

for handler in handlers:
    logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)


class MinorStoppingModel:
    def __init__(self):
        self.r = np.nan
        self.c = np.nan
        self.tau_0 = np.nan
        self.T = np.nan

        self.e = np.nan  # conversion ratio
        self.p = np.nan  # par value per bond
        self.M = np.nan  # initial number of shares
        self.N = np.nan  # initial number of bonds
        self.k = np.nan  # call premium multiplier
        self.d = np.nan  # dividend per share

        self.v_0 = np.nan
        self.sigma = np.nan
        self.nu = np.nan

        self.I = None  # cumulative distribution, must support __getitem__

    @property
    def q(self):
        return self.p / self.e

    def print_scale_summary(self):
        summary = {
            'effective_strike': self.q,
            'initial_asset_per_share': self.v_0,
            'initial_debt_per_share': self.p * self.N / self.M,
            'initial_equity_per_share': self.v_0 - self.p * self.N / self.M
        }
        pprint.pprint(summary)

    @property
    def driver(self):
        """
        This is the dynamics of v, where
        v_t = v_{t-1} + (\nu + \sigma \epsilon_t) v_{t-1} - c_t,
        and where
        c_t = \frac{cpN}{M}(1-I(t)) + d(1 + \frac{eN}{M}I(t)).
        """

        def fn(t, x, noise):
            return x + (self.nu + self.sigma * noise) * x - self.c * self.p * self.N / self.M * (1 - self.I(t)) - self.d * (1 + self.e * self.N / self.M * self.I(t))

        return fn

    @property
    def running_reward(self):
        def fn(t, x):
            return np.power((1 + self.r), -t) * self.c * (t <= (self.tau_0))

        return fn

    @property
    def terminal_reward_call(self):
        def fn(t, x):
            return np.power((1 + self.r), -self.tau_0) * self.k * (t > self.tau_0) * (self.tau_0 <= self.T)

        return fn

    @property
    def terminal_reward_par(self):
        def fn(t, x):
            return np.power((1 + self.r), -self.T) * (t == (self.T + 1)) * (self.tau_0 == (self.T + 1))

        return fn

    @property
    def terminal_reward_conversion(self):
        def fn(t, x):
            return np.power((1 + self.r), -t) / self.q * (x - self.p * self.N / self.M * (1 - self.I(t))) * (1 - self.e * self.N / self.M * self.I(t)) * (t <= self.tau_0 and t <= self.T)

        return fn

    @property
    def terminal_reward(self):
        def fn(t, x):
            return self.p * ((self.terminal_reward_call)(t, x) + (self.terminal_reward_par)(t, x) + (self.terminal_reward_conversion)(t, x))

        return fn

    def get_model(self):
        model = DiscreteStoppingModel()
        model.driver = self.driver
        model.running_reward = self.running_reward
        model.terminal_reward = self.terminal_reward
        return model


if __name__ == '__main__':
    import utils.distribution

    minor_model = MinorStoppingModel()

    minor_model.r = 0.01
    minor_model.c = 0.02
    minor_model.tau_0 = 7
    minor_model.T = 10


    def cdf(t):
        return min(t, minor_model.T) / minor_model.T


    minor_model.e = 10
    minor_model.p = 1000
    minor_model.M = 1e6
    minor_model.N = 1e4
    minor_model.k = 1.2
    minor_model.d = 1
    minor_model.nu = 0
    minor_model.sigma = 1

    minor_model.v_0 = 100

    minor_model.I = cdf

    minor_model.print_scale_summary()

    stopping_model = minor_model.get_model()

    config = DiscreteStoppingConfig()
    config.time_num_grids = 100
    config.time_upper_bound = minor_model.T
    config.time_lower_bound = 0

    config.state_num_grids = 100
    config.state_upper_bound = minor_model.v_0 * 2
    config.state_lower_bound = minor_model.v_0 / 2

    solver = DiscreteStoppingSolver(stopping_model, config)

    solver.solve()
    solver.plot_value_surface()
    solver.plot_stop_flag()
