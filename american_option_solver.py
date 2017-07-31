from solver.discrete_stopping_solver import *


class AmericanOptionModel(DiscreteStoppingModel):
    def __init__(self, div_times, div_amount, r, sigma, strike):
        super().__init__()
        self.div_times = div_times
        self.div_amount = div_amount
        self.r = r
        self.sigma = sigma
        self.strike = strike

    @property
    def div_time_discrete(self):
        return [int(div_time / self.dt) * self.dt for div_time in self.div_times]

    @property
    def driver(self):
        def fn(t, x, noise):
            if t in self.div_time_discrete:
                x -= self.div_amount
            return x * (1 + self.r * self.dt + self.sigma * np.sqrt(self.dt) * noise)

        return fn

    @property
    def terminal_reward(self):
        def fn(t, x):
            return np.exp(-r * t) * max(self.strike - x, 0)

        return fn


def put_payoff(t, x, strike, r):
    return np.exp(-r * t) * max(strike - x, 0)


def put_payoff_getter(strike, r):
    def payoff(t, x):
        return put_payoff(t, x, strike, r)

    return payoff


def call_payoff(t, x, strike, r):
    return np.exp(-r * t) * max(x - strike, 0)


def call_payoff_getter(strike, r):
    def payoff(t, x):
        return call_payoff(t, x, strike, r)

    return payoff


if __name__ == '__main__':
    model = DiscreteStoppingModel()

    # div_times = [0.3, 0.6, 0.9]
    div_times = []
    div_amount = 1
    r = 0.01
    sigma = 0.8
    strike = 100

    model = AmericanOptionModel(div_times=div_times, div_amount=div_amount, r=r, sigma=sigma, strike=strike)

    model.time_num_grids = 50
    model.time_upper_bound = 1
    model.time_lower_bound = 0

    model.state_num_grids = 50
    model.state_upper_bound = 200
    model.state_lower_bound = 50

    solver = DiscreteStoppingSolver(model)

    solver.solve()
    solver.plot_value_surface()
    stopping_distribution = solver.estimate_stopping_distribution(initial_value=100, num_samples=5000)
    stopping_distribution.plot_cdf()
    stopping_distribution.plot_histogram()
    solver.plot_stop_flag()
