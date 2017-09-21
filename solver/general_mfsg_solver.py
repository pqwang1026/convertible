import numpy as np
import pandas as pd
import utils.perf as perf
import matplotlib.pyplot as plt
import logging
import utils.distribution
from solver.discrete_stopping_solver import DiscreteStoppingModel, DiscreteStoppingSolver

logger = logging.getLogger(__name__)


class MFSGModel(DiscreteStoppingModel):
    """
    This MFSG stores the cdf in the member I. When I is fixed, MFSGModel is just a DiscreteStoppingModel.
    """

    def __init__(self):
        super().__init__()
        self.nu = None

    @property
    def mfsg_driver(self):
        raise NotImplementedError

    @property
    def mfsg_terminal_reward(self):
        return lambda t, x, i: 0

    @property
    def mfsg_running_reward(self):
        return lambda t, x, i: 0

    @property
    def driver(self):
        def func(t, x, noise):
            return self.mfsg_driver(t, x, self.nu(t), noise)

        return func

    @property
    def terminal_reward(self):
        def func(t, x):
            return self.mfsg_terminal_reward(t, x, self.nu(t))

        return func

    @property
    def running_reward(self):
        def func(t, x):
            return self.mfsg_running_reward(t, x, self.nu(t))

        return func


class MFSGSolver:
    def __init__(self, model: MFSGModel):
        self.model = model

    def solve_given_nu(self, nu: utils.distribution.Distribution):
        init_value = 0

        self.model.nu = nu

        stopping_solver = DiscreteStoppingSolver(model=self.model)
        stopping_solver.solve()
        stopping_dist = stopping_solver.estimate_stopping_distribution(initial_value=init_value, num_samples=int(1e3))

        stopping_solver.plot_stop_flag()
        return stopping_dist

    def solve(self, init_nu):
        nu = init_nu
        while True:
            nu = self.solve_given_nu(nu)
