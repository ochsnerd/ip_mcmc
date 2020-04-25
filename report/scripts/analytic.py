import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
                     AnalyticAccepter, CountedAccepter,
                     StandardRWProposer,
                     GaussianDistribution)


class Bimodal():
    def __init__(self, mean, variance, l):
        self.left = GaussianDistribution(mean=-mean, covariance=variance)
        self.right = GaussianDistribution(mean=mean, covariance=variance)
        self.l = l

    def __call__(self, x):
        if abs(x) > self.l:
            return 0

        # since we cut the tails, the normalisation is perfect
        return .5 * (self.left(x) + self.right(x))


def normal(x):
    return np.exp(-0.5*x**2) / np.sqrt(2 * np.pi)


def create_StandardRWSampler(density):
    random_walk_proposer = StandardRWProposer(0.25, 1)
    acceptance = AnalyticAccepter(density)
    return MCMCSampler(random_walk_proposer, acceptance, np.random.default_rng(1))


def create_density_plot(sampler, density):
    # TODO: actually figure out how to work with matplotlib objects
    x_0 = np.array([0])

    X = sampler.run(x_0, 500)
    plt.hist(X, bins=20, density=True)
    plt.plot(np.linspace(-10, 10), [density(x) for x in np.linspace(-10, 10)])


def create_autocorrelation_plot(sampler):
    x_0 = np.array([0])

    X = sampler.run(x_0, n_samples=1000, sample_interval=1).flatten()
    ac = np.correlate(X, X, mode='full')
    ac = ac[ac.size//2:]
    ac /= ac[0]

    plt.plot(ac)


def main():
    bimodal_density = Bimodal(3, 1, 10)

    bimodal_RWSampler = create_StandardRWSampler(bimodal_density)

    create_density_plot(bimodal_RWSampler, bimodal_density)
    plt.savefig("../figures/bimodal_density.svg", format='svg')
    plt.clf()

    create_autocorrelation_plot(bimodal_RWSampler)
    plt.savefig("../figures/bimodal_ac.svg", format='svg')
    plt.clf()

    normal_RWSampler = create_StandardRWSampler(normal)

    create_density_plot(normal_RWSampler, normal)
    plt.savefig("../figures/normal_density.svg", format='svg')
    plt.clf()

    create_autocorrelation_plot(normal_RWSampler)
    plt.savefig("../figures/normal_ac.svg", format='svg')
    plt.clf()


if __name__ == '__main__':
    main()
