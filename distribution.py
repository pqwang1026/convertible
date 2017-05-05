import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class EmpiricalDistribution:
    def __init__(self, data):
        self.data = data
        self.data.sort()
        self.data_size = len(data)

    def distplot(self, **kwargs):
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


if __name__ == '__main__':
    dist = DiscreteDistribution([0.1, 0.1, 0.2, 0.6])
    print(dist[2])
    dist.plot_cdf()
