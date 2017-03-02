from OptimalStoppingSolver import ConvertibleModel, OptimalStoppingSolver
import numpy as np


class MeanFieldGameSolver(object):
    def __init__(self, stopping_solver, initial_dist, precision, max_iter):
        self.stopping_solver = stopping_solver
        self.precision = precision
        self.max_iter = max_iter
        self.set_stopping_distribution(initial_dist)
        self.num_iter = 0
        self.continue_flag = 1
        self.error = 10000.0

    def set_stopping_distribution(self, mu):
        self.stopping_solver.model.mu = mu

    def get_stopping_distribution(self):
        return self.stopping_solver.stopping_distribution

    def update(self):
        self.last_dist = self.stopping_solver.model.mu
        self.stopping_solver.solve_full()
        self.new_dist = self.get_stopping_distribution()
        self.set_stopping_distribution(self.new_dist)
        self.num_iter = self.num_iter + 1
        self.error = np.linalg.norm(self.last_dist - self.new_dist)
        print self.error
        print self.new_dist
        if self.num_iter == self.max_iter or self.error <= self.precision:
            self.continue_flag = 0

    def solve(self):
        while self.continue_flag == 1:
            self.update()