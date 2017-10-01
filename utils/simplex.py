import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

sns.set_style('whitegrid')


class SimplexPoint(list):
    def __init__(self, weights):
        assert (np.abs(sum(weights) - 1) < 1e-4)
        assert (all([w >= 0 for w in weights]))
        for weight in weights:
            self.append(weight)
        self.n = len(weights) - 1

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar([i for i in range(0, self.n + 1)], self)


class Triangle:
    """
    A triangle is uniquely determined by a founding vertex and a permutation.
    """

    def __init__(self, v: SimplexPoint, permutation, k):
        self.v = v
        self.permutation = permutation
        self.k = k

    def get_all_vertices(self):
        U = get_U(self.v.n)
        triangle_vertices = [np.array(self.v).astype(float)]
        for i, j in enumerate(self.permutation):
            triangle_vertices.append(triangle_vertices[i] + U[:, j] / self.k)
        triangle_vertices = [SimplexPoint(v) for v in triangle_vertices]
        return triangle_vertices


def get_U(n):
    U = np.zeros((n + 1, n))
    for i in range(0, n):
        U[i][i] = -1
    for i in range(1, n + 1):
        U[i][i - 1] = 1
    return U


def get_covering_triangle(s: SimplexPoint, k: int):
    """
    Given a point in the standard n-simplex, calculate the covering triangle in the k-triangulation.
    """
    print('finding covering triangle for {}'.format(s))
    Q = np.array(s) * k
    prev_lamd = 0
    X = []
    Lamb = []
    for q in Q:
        x_raw = q - prev_lamd
        if np.abs(x_raw - int(round(x_raw))) < 1e-4:
            x = int(round(x_raw))
        else:
            x = np.ceil(x_raw)
        lamb = x - q + prev_lamd
        prev_lamd = lamb
        X.append(x)
        Lamb.append(lamb)
    # sort Lamb, and obtain the permutation to generate the triangle
    Lamb = Lamb[:-1]
    Lamb_ = [(i, lamb) for i, lamb in enumerate(Lamb)]
    sorted(Lamb_, key=lambda x: x[1], reverse=True)
    permutation = [(lambda x: x[0])(x) for x in sorted(Lamb_, key=lambda x: x[1], reverse=True)]

    return Triangle(SimplexPoint(np.array(X) / k), permutation, k)


def get_random_simplex_point(n):
    nodes = [random.uniform(0, 1) for i in range(0, n)]
    nodes = sorted(nodes)
    nodes = [0] + nodes + [1]
    return SimplexPoint(np.diff(nodes))


if __name__ == '__main__':
    import pprint

    sp = SimplexPoint([0.3, 0.2, 0.5])
    triangle = get_covering_triangle(sp, 13)

    print(triangle.get_all_vertices())

    # A = np.zeros(shape=(3, 3))
    # for i, x in enumerate(X):
    #     A[:, i] = x
    # b = np.array(sp)
    # print(np.linalg.solve(A, b))
