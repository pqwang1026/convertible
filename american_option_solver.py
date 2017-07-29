from solver.discrete_stopping_solver import *


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
    config = DiscreteStoppingConfig()

    strike = 100
    r = 0.01
    T = 1
    sigma = 0.2
    div_times = [0.3, 0.6, 0.9]
    div_amount = 10

    config.time_num_grids = 50
    config.time_upper_bound = T
    config.time_lower_bound = 0

    config.state_num_grids = 50
    config.state_upper_bound = 200
    config.state_lower_bound = 50

    dt = T / config.time_num_grids
    div_time_discrete = [int(div_time / dt) * dt for div_time in div_times]


    def driver(t, x, noise):
        if t in div_time_discrete:
            x -= div_amount
        return x * (1 + r * dt + sigma * np.sqrt(dt) * noise)


    model.driver = driver
    model.terminal_reward = put_payoff_getter(100, 0.001)
    # model.terminal_reward = call_payoff_getter(100, 0.001)

    solver = DiscreteStoppingSolver(model, config)

    solver.solve()
    solver.plot_value_surface()
    solver.plot_stop_flag()
