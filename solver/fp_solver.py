import numpy as np
import utils.simplex as splx
import logging
import utils.perf as perf

logger = logging.getLogger(__name__)


class FpSolver:
    @property
    def logger(self):
        return logging.getLogger('.'.join([__name__, self.__class__.__name__]))


class IterataiveSolver(FpSolver):
    def __init__(self, f, n):
        self.f = f
        self.n = n

        self.tol = 1e-3

        self.fixed_point = None
        self.max_iter = 1000

    @perf.timed
    def solve(self):
        x = splx.get_random_simplex_point(self.n)

        cnt = 1

        success = False
        while cnt < self.max_iter:
            self.logger.info('Iteration {0}, current location {1}.'.format(cnt, x))
            cnt += 1
            prev_x = x
            x = self.f(x)

            if np.linalg.norm(np.array(x) - np.array(prev_x)) < self.tol:
                success = True
                break
        if success:
            self.fixed_point = x
        return self.fixed_point


class KakutaniSolver(FpSolver):
    def __init__(self, f, n):
        self.f = f  # f should be a function, that given a n-simplex, it returns a list of n-simplex, i.e. it's a correspondence on n-simplexes
        self.n = n

        self.b = None
        self.b = b = np.array([0.2, 0.3, -0.5] + [0 for _ in range(0, self.n - 2)])
        self.d = np.array(list(b) + [1])
        self.d.shape = (self.n + 2, 1)

        # state of the algorithm
        self.triangle = None
        self.non_basic_idx = None
        self.L = None
        self.weights = None
        self.slack_weight = None

        self.fixed_point = None

    @property
    def all_weights(self):
        return list(self.weights) + [self.slack_weight]

    def move(self):
        """
        Given the triangle and the non-basic index, move to the adjacent triangle.
        """
        self.logger.info('Moving to the next triangle...')
        U = splx.get_U(self.n)
        if self.non_basic_idx == 0:
            new_v = splx.SimplexPoint(np.array(self.triangle.v) + 1 / self.triangle.k * U[:, self.triangle.permutation[0]])
            new_permutation = self.triangle.permutation[1:] + [self.triangle.permutation[0]]
            self.triangle = splx.Triangle(new_v, new_permutation, self.triangle.k)

            self.non_basic_idx = self.n
            non_basic_vertex = splx.SimplexPoint(self.triangle.get_all_vertices()[self.non_basic_idx])
            l = np.array(list(np.array(self.f(non_basic_vertex)) - np.array(non_basic_vertex)) + [1])
            l.shape = (self.n + 2, 1)
            self.L = np.delete(self.L, 0, axis=1)
            self.L = np.concatenate([self.L, l], axis=1)

            self.weights = list(np.delete(self.weights, 0))
            self.weights = self.weights + [0]
        elif self.non_basic_idx == self.n:
            new_v = splx.SimplexPoint(np.array(self.triangle.v) - 1 / self.triangle.k * U[:, self.triangle.permutation[-1]])
            new_permutation = [self.triangle.permutation[-1]] + self.triangle.permutation[:(len(self.triangle.permutation) - 1)]
            self.triangle = splx.Triangle(new_v, new_permutation, self.triangle.k)

            self.non_basic_idx = 0
            non_basic_vertex = splx.SimplexPoint(self.triangle.get_all_vertices()[self.non_basic_idx])
            l = np.array(list(np.array(self.f(non_basic_vertex)) - np.array(non_basic_vertex)) + [1])
            l.shape = (self.n + 2, 1)
            self.L = np.delete(self.L, self.n, axis=1)
            self.L = np.concatenate([l, self.L], axis=1)

            self.weights = list(np.delete(self.weights, self.n))
            self.weights = [0] + self.weights
        else:
            new_v = self.triangle.v
            new_permutation = self.triangle.permutation
            front = self.triangle.permutation[self.non_basic_idx - 1]
            back = self.triangle.permutation[self.non_basic_idx]
            new_permutation[self.non_basic_idx - 1] = back
            new_permutation[self.non_basic_idx] = front
            self.triangle = splx.Triangle(new_v, new_permutation, self.triangle.k)
            non_basic_vertex = splx.SimplexPoint(self.triangle.get_all_vertices()[self.non_basic_idx])
            l = list(np.array(self.f(non_basic_vertex)) - np.array(non_basic_vertex)) + [1]
            self.L[:, self.non_basic_idx] = l

    def pivot(self):
        """
        Given a triangle and a basic feasible solution, pivot the non-basic index into the solution and squeeze one out.
        """
        self.logger.info('Pivoting from one non-basic vertex to another vertex in the triangle...')
        L_ = np.concatenate([self.L, self.d], axis=1)
        ratio = np.linalg.solve(np.delete(np.delete(L_, self.non_basic_idx, axis=1), 0, axis=0),
                                L_[:, self.non_basic_idx][1:])  # if non-basic increase by 1, all the others must decrease by ratio
        ratio = np.insert(ratio, self.non_basic_idx, -1)

        tmp = []
        for i, spike in enumerate(ratio):
            if spike <= 0:
                continue
            tmp.append((i, self.all_weights[i] / spike))
        tmp = sorted(tmp, key=lambda x: x[1])
        new_non_basic_idx = tmp[0][0]
        theta = tmp[0][1]

        # update non-basic index and weights
        new_all_weights = list(np.array(self.all_weights) - theta * ratio)
        self.weights = new_all_weights[:-1]
        self.slack_weight = new_all_weights[-1]
        self.non_basic_idx = new_non_basic_idx

    def initialize(self, k):
        b = self.b
        while True:
            x0 = splx.get_random_simplex_point(self.n)

            tmp = []
            for i, spike in enumerate(b):
                if spike >= 0:
                    continue
                tmp.append(-x0[i] / spike)
            theta = max(tmp)
            x0 = splx.SimplexPoint(x0 + theta * b)

            triangle = splx.get_covering_triangle(x0, k)
            triangle_vertices = triangle.get_all_vertices()
            L = np.zeros(shape=(self.n + 2, self.n + 1))
            for i in range(0, self.n + 1):
                L[:, i] = list(np.array(self.f(splx.SimplexPoint(triangle_vertices[i]))) - np.array(splx.SimplexPoint(triangle_vertices[i]))) + [1]
            d = np.array(list(b) + [1])
            d.shape = (self.n + 2, 1)
            m = np.array([0 for _ in range(0, self.n + 1)] + [1])

            L_ = np.concatenate([L, d], axis=1)
            pivot_idx = None

            for i in [self.n + 1] + list(range(0, self.n + 1)):
                sub_L_ = np.delete(np.delete(L_, i, axis=1), 0, axis=0)
                v = np.linalg.solve(sub_L_, m[1:])
                if all([w >= 0 for w in v]):
                    pivot_idx = i
                    if pivot_idx == self.n + 1:
                        self.slack_weight = 0
                        self.weights = v
                    else:
                        self.slack_weight = v[self.n]
                        self.weights = np.delete(list(np.insert(v, i, 0)), self.n + 1)
                    print(self.weights)
                    break
            if pivot_idx is not None:
                break
        self.triangle = triangle
        self.non_basic_idx = pivot_idx
        self.L_ = L_
        self.L = L
        return triangle_vertices, pivot_idx

    @perf.timed
    def solve(self, k):
        self.initialize(k)
        if self.non_basic_idx == self.n + 1:
            self.logger.info('Found a fixed point.')
            self.fixed_point = 0
            triangle_vertices = self.triangle.get_all_vertices()
            for i, weight in enumerate(self.weights):
                self.fixed_point += weight * np.array(triangle_vertices[i])
            self.fixed_point = splx.SimplexPoint(self.fixed_point)
            return self.fixed_point

        cnt = 1
        while True:
            self.logger.info('Iteration {0}, Current triangle: {1}'.format(cnt, self.triangle.get_all_vertices()))
            self.move()
            self.pivot()
            cnt += 1
            if self.non_basic_idx == self.n + 1:
                break
        self.logger.info('Found a fixed point.')
        self.fixed_point = 0
        triangle_vertices = self.triangle.get_all_vertices()
        for i, weight in enumerate(self.weights):
            self.fixed_point += weight * np.array(triangle_vertices[i])
        self.fixed_point = splx.SimplexPoint(self.fixed_point)
        return self.fixed_point


if __name__ == '__main__':
    def f(s: splx.SimplexPoint):
        """
        For a given 2-simplex point (x0, x1, x2), this function gives (sqrt(x1), 1-sqrt(x1), 0).
        """
        if s.n != 2:
            raise RuntimeError('dimension does not match!')
        return splx.SimplexPoint([np.sqrt(s[1]), 1 - np.sqrt(s[1]), 0])


    def f(s: splx.SimplexPoint):
        """
        This is a rotation.
        """
        if s.n != 2:
            raise RuntimeError('dimension does not match!')
        return splx.SimplexPoint([s[1], s[2], s[0]])


    solver = KakutaniSolver(f, n=2)
    print(solver.solve(k=4))
