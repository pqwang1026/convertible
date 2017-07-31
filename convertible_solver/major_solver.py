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


class MajorStoppingModel(DiscreteStoppingModel):
    def __init__(self):
        super().__init__()
        self.r = np.nan  # interest rate
        self.c = np.nan  # coupon rate
        self.T = np.nan  # terminal time

        self.e = np.nan  # conversion ratio
        self.p = np.nan  # par value per bond
        self.M = np.nan  # initial number of shares
        self.N = np.nan  # initial number of bonds
        self.k = np.nan  # call premium multiplier
        self.d = np.nan  # dividend per share

        self.R_0 = np.nan  # initial refinancing ratg
        self.sigma = np.nan  # volatility of subjective growth rate
        self.nu = np.nan  # drift of growth rate

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
        This is the dynamics of R, where
        R_t = R_{t-1} + \sigma \epsilon_t
        """

        def fn(t, x, noise):
            return x + x * self.sigma * np.sqrt(self.dt) * noise

        return fn

    @property
    def running_reward(self):
        def fn(t, x):
            return np.power((1 + self.r), -t) * (self.c + (self.d / self.q - self.c) * self.I(t))

        return fn

    @property
    def terminal_reward_refinance_div(self):
        def fn(t, x):
            return 1 / self.r * (np.power((1 + self.r), -t) - np.power((1 + self.r), -self.T)) * (self.k * x + (self.d / self.q - self.k * x) * self.I(t))

        return fn

    @property
    def terminal_reward_par(self):
        def fn(t, x):
            return (1 - self.I(self.T)) * (t == (self.T))

        return fn

    @property
    def terminal_reward_refinance_par(self):
        def fn(t, x):
            return self.k * (1 - self.I(t)) * (t < self.T)

        return fn

    @property
    def terminal_reward(self):
        def fn(t, x):
            return self.p * self.N * ((self.terminal_reward_refinance_div)(t, x) + (self.terminal_reward_par)(t, x) + (self.terminal_reward_refinance_par)(t, x))

        return fn


if __name__ == '__main__':
    import utils.distribution

    model = MajorStoppingModel()

    model.r = 0.01
    model.c = 0.02
    model.T = 10


    def cdf(t):
        return min(t, model.T) / model.T


    model.e = 10
    model.p = 1000
    model.M = 1e6
    model.N = 1e4
    model.k = 1.2
    model.d = 1
    model.nu = 0
    model.sigma = 0.0005

    model.R_0 = 0.023

    model.I = cdf

    # model.print_scale_summary()

    model.time_num_grids = 50
    model.time_upper_bound = model.T
    model.time_lower_bound = 0

    model.state_num_grids = 50
    model.state_upper_bound = model.R_0 * 2
    model.state_lower_bound = model.R_0 / 2

    solver = DiscreteStoppingSolver(model)

    solver.solve()
    solver.plot_value_surface()
    solver.plot_stop_flag()
