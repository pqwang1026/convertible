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

        self.R_0 = np.nan  # initial refinancing rate
        self.theta = np.nan  # mean reverting rate of refinancing interest rate
        self.R_lt = np.nan  # long-term refinancing interest rate
        self.sigma = np.nan  # volatility of refinancing interest rate

        self.I = None  # cumulative distribution, must support __call__
        self.optimize_type = StoppingOptimizeType.MINIMIZE
        self.normalize = True

    def update_bounds(self):
        self.time_upper_bound = self.T
        self.time_lower_bound = 0
        up_factor = self.sigma * np.sqrt(self.time_increment)
        dn_factor = self.sigma * np.sqrt(self.time_increment)
        self.state_upper_bound = self.R_0 + up_factor * (self.time_num_grids + 1)
        self.state_lower_bound = self.R_0 - dn_factor * (self.time_num_grids + 1)

        logger.info('Updated self-adaptive state upper and lower bounds, upper bound = {0}, lower bound = {1}.'.format(self.state_upper_bound, self.state_lower_bound))

    @property
    def q(self):
        return self.p / self.e

    def print_scale_summary(self):
        summary = {
            'dividend_per_share': self.d,
            'coupon_rate': self.c,
            'something': self.d / self.q
        }
        pprint.pprint(summary, width=1)

    @property
    def driver(self):
        """
        This is the dynamics of R, where
        R_t = R_{t-1} + \theta ( R_lt - R_{t-1}) + \sigma \epsilon_t
        which is a Vascicek-type mean reverting process.
        """

        def fn(t, x, noise):
            return x + self.theta * (self.R_lt - x) * self.dt + self.sigma * np.sqrt(self.dt) * noise

        return fn

    @property
    def running_reward(self):
        def fn(t, x):
            res = self.p * np.power((1 + self.r), -t) * (self.c + (self.d / self.q - self.c) * self.I(t))
            if not self.normalize:
                res *= self.N
            return res

        return fn

    @property
    def terminal_reward_refinance_div(self):
        def fn(t, x):
            return 1 / self.r * (np.power((1 + self.r), -t) - np.power((1 + self.r), -self.T)) * (self.k * x + (self.d / self.q - self.k * x) * self.I(t))

        return fn

    @property
    def terminal_reward_par(self):
        def fn(t, x):
            return np.power(1 + self.r, -self.T) * (1 - self.I(self.T - self.time_increment)) * (t == (self.T))

        return fn

    @property
    def terminal_reward_refinance_par(self):
        def fn(t, x):
            return np.power(1 + self.r, -self.T) * self.k * (1 - self.I(t)) * (t < self.T)

        return fn

    @property
    def terminal_reward(self):
        def fn(t, x):
            res = self.p * ((self.terminal_reward_refinance_div)(t, x) + (self.terminal_reward_par)(t, x) + (self.terminal_reward_refinance_par)(t, x))
            if not self.normalize:
                res *= self.N
            return res

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
    model.k = 1
    model.d = 5

    model.sigma = 0.005
    model.R_0 = 0.03
    model.R_lt = 0.03
    model.theta = 0.1

    model.I = cdf

    # model.print_scale_summary()

    model.time_num_grids = 10
    model.state_num_grids = 100

    solver = DiscreteStoppingSolver(model)
    model.print_scale_summary()

    solver.solve()

    stopping_dist = solver.estimate_stopping_distribution(initial_value=model.R_0, num_samples=1000)
    stopping_dist.plot_cdf()

    solver.plot_value_surface()
    solver.plot_stop_flag()
