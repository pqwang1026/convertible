import numpy as np
import pandas as pd
import utils.perf as perf
import matplotlib.pyplot as plt
import logging
import utils.distribution
from solver.general_mfsg_solver import MFSGModel, MFSGSolver

logger = logging.getLogger(__name__)


class TestModel(MFSGModel):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    @property
    def mfsg_driver(self):
        def func(t, x, i, noise):
            return x + self.mu * self.dt + self.sigma * np.sqrt(self.dt) * noise

        return func

    @property
    def mfsg_terminal_reward(self):
        return lambda t, x, i: x - i


if __name__ == '__main__':
    T = 1
    mu = 0
    sigma = 1

    model = TestModel(mu=mu, sigma=sigma)
    model.time_num_grids = 40
    model.time_upper_bound = T
    model.time_lower_bound = 0
    model.state_num_grids = 40
    model.state_upper_bound = 3
    model.state_lower_bound = -3

    solver = MFSGSolver(model=model)

    init_nu = utils.distribution.Distribution(nodes=[0, 0.25, 0.5, 0.75, 1], probabilities=[0.2, 0.2, 0.2, 0.2, 0.2])

    solver.solve(init_nu=init_nu)
