from solver.discrete_stopping_solver import *


class OptionType:
    CALL = 'CALL'
    PUT = 'PUT'


class AmericanOptionModel(DiscreteStoppingModel):
    def __init__(self, spot, r, sigma, strike, T, div_times, div_amount, option_type):
        super().__init__()
        self.spot = spot
        self.div_times = div_times
        self.div_amount = div_amount
        self.r = r
        self.sigma = sigma
        self.strike = strike
        self.T = T
        self.option_type = option_type

    def update_bounds(self):
        self.state_upper_bound = self.spot * np.exp((self.r - self.sigma * self.sigma / 2) * self.T + self.sigma * 2 * np.sqrt(self.T))
        self.state_lower_bound = self.spot * np.exp((self.r - self.sigma * self.sigma / 2) * self.T - self.sigma * 2 * np.sqrt(self.T)) - self.div_amount * len(self.div_times)
        self.time_upper_bound = self.T
        self.time_lower_bound = 0

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
        if self.option_type == OptionType.PUT:
            def fn(t, x):
                return np.exp(-r * t) * max(self.strike - x, 0)
        elif self.option_type == OptionType.CALL:
            def fn(t, x):
                return np.exp(-r * t) * max(x - self.strike, 0)
        else:
            raise NotImplementedError

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
    div_amount = 2
    r = 0.01
    sigma = 0.2
    strike = 100
    T = 1
    spot = 100
    option_type = OptionType.PUT

    model = AmericanOptionModel(spot=spot, T=T, div_times=div_times, div_amount=div_amount, r=r, sigma=sigma, strike=strike, option_type=option_type)

    model.time_num_grids = 50
    model.state_num_grids = 100

    solver = DiscreteStoppingSolver(model)

    solver.solve()
    solver.plot_value_surface()
    stopping_distribution = solver.estimate_stopping_distribution(initial_value=model.spot, num_samples=1000)
    stopping_distribution.plot_cdf()
    solver.plot_stop_flag()
