import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
                     AnalyticAccepter, CountedAccepter, StandardRWAccepter, pCNAccepter,
                     StandardRWProposer, pCNProposer,
                     AnalyticPotential,
                     GaussianDistribution)

from helpers import store_figure


class Bimodal():
    def __init__(self, mean, variance, l):
        self.left = GaussianDistribution(mean=-mean, covariance=variance)
        self.right = GaussianDistribution(mean=mean, covariance=variance)
        self.l = l

    def __call__(self, x):
        if abs(x) > self.l:
            return 0

        # since we cut the tails, the normalisation is not perfect
        return .5 * (self.left(x) + self.right(x))


def normal(x):
    return np.exp(-0.5*x**2) / np.sqrt(2 * np.pi)


def create_StandardRWSampler(density):
    mean = np.array([0])
    covariance = np.array([1], ndmin=2)
    sqrt_covariance = np.array([1], ndmin=2)
    prior = GaussianDistribution(mean, covariance)
    potential = AnalyticPotential(posterior=density, prior=prior)

    proposer = StandardRWProposer(delta=0.25,
                                  dims=1,
                                  sqrt_covariance=sqrt_covariance)
    accepter = StandardRWAccepter(potential=potential,
                                  prior=prior)
    return MCMCSampler(proposer, accepter, np.random.default_rng(1))


def create_pCNSampler(density):
    mean = np.array([0])
    covariance = np.array([1], ndmin=2)
    prior = GaussianDistribution(mean, covariance)

    potential = AnalyticPotential(posterior=density, prior=prior)

    # beta != delta of other proposers, but
    # it could easily be translated if someone took the 30s to do it
    proposer = pCNProposer(beta=0.25, prior=prior)
    accepter = pCNAccepter(potential=potential)
    return MCMCSampler(proposer, accepter, np.random.default_rng(1))


def create_AnalyticSampler(density):
    proposer = StandardRWProposer(0.25, 1)
    accepter = AnalyticAccepter(density)
    return MCMCSampler(proposer, accepter, np.random.default_rng(1))


def create_density_plot(sampler, density):
    # TODO: actually figure out how to work with matplotlib objects
    x_0 = np.array([0])

    X = sampler.run(x_0, 500)
    plt.hist(X, bins=20, density=True, label="Sampled")
    x_analytic = np.linspace(-10, 10, 200)
    plt.plot(x_analytic,
             [density(x) for x in x_analytic],
             label="True")
    plt.legend()


def create_autocorrelation_plot(sampler, name, chain_length=1000, nruns=1):
    x_0 = np.array([0])

    ac_avg = np.zeros((chain_length, ))

    for i in range(nruns):
        X = sampler.run(x_0, n_samples=chain_length, sample_interval=1).flatten()
        ac = np.correlate(X, X, mode='full')
        ac_avg += ac[-chain_length:]

    # average and normalize
    ac_avg /= nruns
    ac_avg /= ac_avg[0]

    plt.plot(ac_avg, label=name)


def plot_sampler_characteristics(sampler_generator, density, name):
    sampler = sampler_generator(density)

    create_density_plot(sampler, density)
    plt.title(name + " density sampling")
    store_figure(name + "_density")

    create_autocorrelation_plot(sampler, name)
    plt.title("Autocorrelation of sampling from " + name + " density")
    store_figure(name + "_ac")

    print(name + " done")


def plot_autocorrelations(density, sampler_generators, names, nruns, chain_length):
    for sampler_generator, name in zip(sampler_generators, names):
        create_autocorrelation_plot(sampler_generator(density),
                                    name,
                                    chain_length,
                                    nruns)

        print(name + " done")

    plt.title(f"Autocorrelation averaged over {nruns} runs")
    plt.legend()

    store_figure("_".join(names + [str(nruns)]))


def main():
    bimodal_density = Bimodal(3, 1, 10)

    plot_autocorrelations(bimodal_density,
                          [create_AnalyticSampler, create_StandardRWSampler, create_pCNSampler],
                          ["analytic", "standard_rw", "pCN"],
                          nruns=50,
                          chain_length=5000)

    # plot_sampler_characteristics(create_StandardRWSampler, bimodal_density, "standard_bimodal")

    # plot_sampler_characteristics(create_StandardRWSampler, normal, "standard_normal")

    # plot_sampler_characteristics(create_pCNSampler, bimodal_density, "pCN_bimodal")

    # plot_sampler_characteristics(create_pCNSampler, normal, "pCN_normal")

    # plot_sampler_characteristics(create_AnalyticSampler, bimodal_density, "analytic_bimodal")

    # plot_sampler_characteristics(create_AnalyticSampler, normal, "analytic_normal")


if __name__ == '__main__':
    main()
