from solver.discrete_stopping_solver import *
import pprint
import utils.perf as perf
import logging
import utils.distribution

logger = logging.getLogger(__name__)

stream_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s> %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.DEBUG)

handlers = [stream_handler]

for handler in handlers:
    logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)


class MinorStoppingModel(DiscreteStoppingModel):
    def __init__(self):
        super().__init__()
        self.r = np.nan  # interest rate
        self.c = np.nan  # coupon rate

        self.tau_0 = np.nan  # major player's stpping time
        self.major_stopping_dist = None

        self.T = np.nan  # terminal time

        self.e = np.nan  # conversion ratio
        self.p = np.nan  # par value per bond
        self.M = np.nan  # initial number of shares
        self.N = np.nan  # initial number of bonds
        self.k = np.nan  # call premium multiplier
        self.d = np.nan  # dividend per share

        self.v_0 = np.nan  # initial firm total asset value
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
        This is the dynamics of v, where
        v_t = v_{t-1} + (\nu + \sigma \epsilon_t) v_{t-1} - c_t,
        and where
        c_t = \frac{cpN}{M}(1-I(t)) + d(1 + \frac{eN}{M}I(t)).
        """

        def fn(t, x, noise):
            res = x + (self.nu + self.sigma * np.sqrt(self.dt) * noise) * x - self.c * self.p * self.N / self.M * (1 - self.I(t)) - self.d * (1 + self.e * self.N / self.M * self.I(t))
            return res

        return fn

    @property
    def running_reward(self):
        def fn(t, x):
            # return np.power((1 + self.r), -t) * self.c * (t <= (self.tau_0))
            return np.power((1 + self.r), -t) * self.c * (1 - self.major_stopping_dist(t - self.time_increment))

        return fn

    @property
    def terminal_reward_call(self):
        def fn(t, x):
            # return np.power((1 + self.r), -self.tau_0) * self.k * (t > self.tau_0) * (self.tau_0 < self.T)
            def discount(s):
                return np.power((1 + self.r), -s) * (s < min(t, self.T))

            res = self.k * (discount * self.major_stopping_dist)
            return res

        return fn

    @property
    def terminal_reward_par(self):
        def fn(t, x):
            # return np.power((1 + self.r), -self.T) * (t == (self.T)) * (self.tau_0 == (self.T))
            res = np.power((1 + self.r), -self.T) * (t == (self.T)) * self.major_stopping_dist.pdf_eval(self.T)
            return res

        return fn

    @property
    def terminal_reward_conversion(self):
        def fn(t, x):
            # return np.power((1 + self.r), -t) / self.q * (x - self.p * self.N / self.M * (1 - self.I(t))) * (1 - self.e * self.N / self.M * self.I(t)) * (t <= self.tau_0 and t <= self.T)
            res = np.power((1 + self.r), -t) / self.q * (x - self.p * self.N / self.M * (1 - self.I(t))) * (1 - self.e * self.N / self.M * self.I(t)) * (t <= self.T) * (
                1 - self.major_stopping_dist(t - self.time_increment))
            return res

        return fn

    @property
    def terminal_reward(self):
        def fn(t, x):
            return self.p * ((self.terminal_reward_call)(t, x) + (self.terminal_reward_par)(t, x) + (self.terminal_reward_conversion)(t, x))

        return fn


if __name__ == '__main__':
    import utils.distribution

    model = MinorStoppingModel()

    model.r = 0.01
    model.c = 0.1

    # model.major_stopping_dist = utils.distribution.SampleDistribution(data=[9])

    model.T = 10

    model.e = 10
    model.p = 1000
    model.M = 1e6
    model.N = 1e4
    model.k = 1.2
    model.d = 1
    model.nu = 0
    model.sigma = 0.4

    model.v_0 = 100

    model.print_scale_summary()

    model.time_num_grids = 50
    model.time_upper_bound = model.T
    model.time_lower_bound = 0

    model.state_num_grids = 50
    model.state_upper_bound = model.v_0 * 2
    model.state_lower_bound = model.v_0 / 2

    model.major_stopping_dist = utils.distribution.Distribution([i * model.T / model.time_num_grids for i in range(1, model.time_num_grids + 1)],
                                                                [1 / model.time_num_grids for _ in range(1, model.time_num_grids + 1)])
    # model.major_stopping_dist = utils.distribution.SampleDistribution(data=[9])
    model.I = model.major_stopping_dist

    solver = DiscreteStoppingSolver(model)

    solver.solve()
    solver.plot_value_surface()

    stopping_distribution = solver.estimate_stopping_distribution(initial_value=model.v_0, num_samples=5000)
    stopping_distribution.plot_cdf()

    solver.plot_stop_flag()
