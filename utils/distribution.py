import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


class EmpiricalDistribution:
    def __init__(self, data):
        self.data = data
        self.data.sort()
        self.data_size = len(data)

    def plot_histogram(self, **kwargs):
        sns.distplot(self.data, **kwargs)

    def get_sample_pdf_series(self):
        counter = dict()
        for datum in self.data:
            if datum not in counter:
                counter[datum] = 1
            else:
                counter[datum] += 1

        data_uniq = list(set(self.data))
        data_uniq.sort()
        counter_list = []
        for datum in data_uniq:
            counter_list.append(counter[datum])

        return pd.Series(data=counter_list, index=data_uniq) / len(self.data)

    def get_sample_cdf_series(self):
        pdf = self.get_sample_pdf_series()
        return pdf.cumsum()

    def get_sample_cdf_fn(self):
        cdf_series = self.get_sample_cdf_series()

        def cdf(x):
            return np.interp(x, cdf_series.index, cdf_series.data)

        return cdf

    def plot_cdf(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cdf = self.get_sample_cdf_series()
        ax.step(cdf.index, cdf.data, where='post')
        plt.show()


class DiscreteDistribution:
    """
    This is a distribution object that's supported on {0, 1, ..., T}.
    """

    def __init__(self, pdf):
        self.T = len(pdf) - 1
        self.pdf = np.array(pdf)
        assert ((sum(pdf) - 1) < 1e-5)

    @property
    def cdf(self):
        return np.cumsum(self.pdf)

    @classmethod
    def get_uniform_dist(cls, T):
        pdf = [1 / (T + 1) for i in range(0, T + 1)]
        return DiscreteDistribution(pdf)

    def plot_cdf(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cdf = pd.Series(self.cdf)
        ax.step(cdf.index, cdf.values, where='post')
        plt.show()

    def __getitem__(self, x):
        return self.pdf[x]


class Distribution:
    def __init__(self, nodes, probabilities):
        assert len(set(nodes)) == len(nodes)
        self.pdf = pd.Series(probabilities, nodes)
        self.pdf.sort_index(inplace=True)

        self.cdf = self.pdf.cumsum()
        self.cdf.sort_index(inplace=True)
        self.cdf[self.cdf.index[0] - 1e-5] = 0
        self.cdf.sort_index(inplace=True)

        assert ((sum(self.pdf) - 1) < 1e-5)

    def pdf_eval(self, x):
        if x in self.pdf.index:
            return self.pdf[x]
        else:
            return 0

    def cdf_eval(self, x):
        return np.interp(x, self.cdf.index, self.cdf.values)

    def plot_cdf(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cdf = self.cdf
        ax.step(cdf.index, cdf.data, where='post', marker='o', markersize=5)

    def __call__(self, x):
        """
        The call method is designed to access the cdf of the distribution.
        """
        return self.cdf_eval(x)

    def __rmul__(self, f):
        """
        This is the integral of f with respect to this distribution.
        """
        sum = 0
        for node, probability in self.pdf.iteritems():
            sum += f(node) * probability
        return sum

    def __add__(self, other):
        nodes = list(self.pdf.index) + list(other.pdf.index)
        probabilities = list(self.pdf.values) + list(other.pdf.values)
        return Distribution(nodes, probabilities)

    def mult(self, a):
        return Distribution(list(self.pdf.index), list(np.array(self.pdf.values * a)))


class SampleDistribution(Distribution):
    def __init__(self, data):
        counter = dict()
        for datum in data:
            if datum not in counter:
                counter[datum] = 1
            else:
                counter[datum] += 1

        data_uniq = list(set(data))
        data_uniq.sort()
        counter_list = []
        for datum in data_uniq:
            counter_list.append(counter[datum])

        pdf = pd.Series(data=counter_list, index=data_uniq) / len(data)
        super().__init__(list(pdf.index), list(pdf.values))


def generate_simplex_sample(p):
    nodes = [random.uniform(0, 1) for i in range(0, p)]
    nodes = sorted(nodes)
    nodes = [0] + nodes + [1]
    return np.diff(nodes)


if __name__ == '__main__':
    data = [9]
    dist = SampleDistribution(data)
    dist.plot_cdf()
    plt.show()
    pass
