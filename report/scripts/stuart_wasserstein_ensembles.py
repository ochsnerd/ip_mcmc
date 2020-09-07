from sys import exit

import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from ip_mcmc import (EnsembleManager,
                     MCMCSampler,
                     StandardRWAccepter,
                     ConstStepStandardRWProposer,
                     VarStepStandardRWProposer,
                     EvolutionPotential,
                     GaussianDistribution)

from helpers import (DATA_DIR,
                     store_figure,
                     autocorrelation,
                     wasserstein_distance)


class SyntheticModel:
    """
    Model based on eq 2.5 in Stuart 2010:
    y = G(u) + eta
    with eta ~ noise

    noise: GaussianDistribution
    """
    def __init__(self, observation_operator, noise):
        self.G = observation_operator
        self.eta = noise

    def evolve(self, u):
        """
        Evolve the state u according to the model equation
        """
        return self.G(u)

    def observe(self, u, rng):
        """
        Do one observation of the model, ie.
        y = self.evolve(u),
        """
        self.y = self.G(u) + self.eta.sample(rng)
        return self.y

    def noise_pdf(self, x):
        return self.eta(x)


class PWLinear:
    """linearly decrease delta until burn_in is finished, then keep it constant"""
    def __init__(self, start_delta, end_delta, len_burn_in):
        self.d_s = start_delta
        self.d_e = end_delta
        self.l = len_burn_in

        self.slope = (start_delta - end_delta) / len_burn_in

    def __call__(self, i):
        if i > self.l:
            return self.d_e
        return self.d_s - self.slope * i

    def __repr__(self):
        """For filename"""
        return f"pwl_{self.d_s}_{self.d_e}_{self.l}"


def build_evolution_RW_sampler(observation_operator,
                               data,
                               prior,
                               noise):
    proposer = ConstStepStandardRWProposer(delta=0.05, prior=prior)
    # delta = PWLinear(0.1, 0.001, 250)
    # proposer = VarStepStandardRWProposer(delta=delta, prior=prior)
    potential = EvolutionPotential(observation_operator, data, noise)
    accepter = StandardRWAccepter(potential, prior)
    return MCMCSampler(proposer, accepter)


def create_data(ensemble_size,
                chain_lengths,
                observation_operator,
                dim_u, ground_truth,
                filename):

    prior_std_dev = 0.5
    prior_covariance = prior_std_dev**2 * np.identity(dim_u)
    prior = GaussianDistribution(mean=np.zeros((dim_u)),
                                 covariance=prior_covariance)

    gamma = 0.1
    # hack to get dim of noise
    n = len(observation_operator(ground_truth))
    noise_covariance = gamma**2 * np.identity(n)
    noise = GaussianDistribution(mean=np.zeros((n)),
                                 covariance=noise_covariance)

    model = SyntheticModel(observation_operator=observation_operator,
                           noise=noise)

    rngs = [np.random.default_rng(i) for i in range(ensemble_size)]

    y = model.observe(ground_truth, rngs[0])

    print(f"y = {y}")

    # Compute chains
    sampler = build_evolution_RW_sampler(observation_operator=observation_operator,
                                         data=y,
                                         prior=prior,
                                         noise=noise)

    u_0 = np.zeros(dim_u)
    burn_in = 500
    sample_interval = 15
    ensembles = []
    for n_samples in chain_lengths:
        chain_start = partial(sampler.run, u_0, n_samples, burn_in, sample_interval)

        ensemble_manager = EnsembleManager(DATA_DIR,
                                           f"{filename}_E={ensemble_size}_N={n_samples}")

        ensemble = ensemble_manager.compute(chain_start,
                                            rngs,
                                            ensemble_size)

        ensembles.append(ensemble)

    # Compute reference
    ref_n_samples = int(1e7)
    ref_chain_start = partial(sampler.run,
                              u_0,
                              ref_n_samples,
                              burn_in,
                              sample_interval)

    ref_manager = EnsembleManager(DATA_DIR,
                                  f"{filename}_ref_N={ref_n_samples}")

    ref_chain = ref_manager.compute(ref_chain_start, rngs[:1], 1)[0, :, :]

    # Since we have zero-centered priors, we don't have to add the means

    for ensemble in ensembles:
        show_ensemble(ensemble, ref_chain, prior, ground_truth)

    return ensembles, ref_chain


def wasserstein_convergence(ensembles, ref_chain, filename):
    # only 1D u for now
    n_ensembles = len(ensembles)
    ensemble_size = ensembles[0].shape[0]
    for ensemble in ensembles:
        assert ensemble_size == ensemble.shape[0], "Require equal-sized ensembles"

    u_range = np.array([np.min([np.min(ensemble) for ensemble in ensembles]),
                        np.max([np.max(ensemble) for ensemble in ensembles])])

    print(f"All values are between {u_range[0]} and {u_range[1]}")

    n_bins = 20
    ref_binned = np.histogram(ref_chain[0, :],
                              bins=n_bins,
                              range=u_range,
                              density=False)[0]
    # Hand-made "normalization"
    ref_binned = ref_binned / np.sum(ref_binned)

    distances = np.zeros((n_ensembles, ensemble_size))

    print(type(ensembles[0]))
    for j, ensemble in enumerate(ensembles):
        print(f"Working at {j}th ensemble, with chain-length {ensemble.shape[2]}")
        print(f"Whole ensemble:")
        print(f"mean: {np.mean(ensemble)}")
        print(f"variance: {np.var(ensemble)}")
        print(f"First chain:")
        print(f"mean: {np.mean(ensemble[0, 0, :])}")
        print(f"variance: {np.var(ensemble[0, 0, :])}")

        # Bin ensembles
        ensemble_binned = np.empty((ensemble_size, n_bins))
        for i in range(ensemble_size):
            ensemble_binned[i, :] = np.histogram(ensemble[i, :],
                                                 bins=n_bins,
                                                 range=u_range,
                                                 density=False)[0]
            ensemble_binned[i, :] = ensemble_binned[i, :] / np.sum(ensemble_binned[i, :])

            print(f"Binned ensemble {i}, shape: {ensemble_binned[i, :].shape}, norm: {np.sum(ensemble_binned[i, :])}")

        # Compute distance to reference
        for i in range(ensemble_size):
            distances[j, i] = wasserstein_distance(ensemble_binned[i, :],
                                                   ref_binned,
                                                   u_range.reshape(1,2))

    for j, ensemble in enumerate(ensembles):
        print(f"Looking at the {j}th ensemble, with chain-length {ensemble.shape[2]}")
        print(f"{np.mean(distances[j, :])}")
        print(f"{np.var(distances[j, :])}")

    chain_lengths = [ensemble.shape[2] for ensemble in ensembles]
    means = [np.mean(distances[j, :]) for j in range(n_ensembles)]
    l_quartile = [np.quantile(distances[j, :], 0.25) for j in range(n_ensembles)]
    u_quartile = [np.quantile(distances[j, :], 0.75) for j in range(n_ensembles)]

    plt.plot(chain_lengths, means, label="mean")
    plt.plot(chain_lengths, l_quartile, label="lower quartile")
    plt.plot(chain_lengths, u_quartile, label="upper quartile")
    plt.plot(chain_lengths, [np.sqrt(chain_lengths[0]) * means[0] / np.sqrt(a) for a in chain_lengths], '--', label="O(L^(-1/2))")
    plt.title("$W_1$ for different chain lengths")
    plt.xlabel("Length of the chain")
    plt.xscale("log")
    plt.ylabel("$W_1$")
    plt.legend()
    store_figure(f"{filename}_wasserstein_convergence_chain")


def show_ensemble(ensemble, ref_chain, prior, ground_truth):
    dim_u = ref_chain.shape[0]
    ensemble_size = ensemble.shape[0]
    for i in range(dim_u):       
        for j in range(ensemble_size):
            plt.plot(ensemble[j, i, :])
        plt.title(f"u[{i}]")
        plt.show()

    # for i in range(dim_u):
    #     plt.plot(ref_chain[i, :], label=f"ref u[{i}]")
    # plt.title("Reference chain")
    # plt.legend()
    # plt.show()

    x_vals = np.linspace(-1, 3, 500)
    prior_vals = np.array([prior(x) for x in x_vals])

    plt.plot(x_vals, prior_vals, label="prior")
    for j in range(ensemble_size):
        plt.hist(ensemble[j, 0, :], bins=50, density=True, alpha=0.5)
    plt.hist(ref_chain[0, :], bins=50, density=True, label="Reference chain")
    plt.axvline(ground_truth, c='r')
    plt.title(f"Prior and Posterior Ensemble")
    plt.legend()
    plt.show()


def G_21(u):
    return np.array([np.dot(np.array([1]), u)])


def G_22(u):
    return np.array([1, 2, 3, 4]) * (u + 0.5 * u**3)


def main():
    print("Stuart example 21")
    ground_truth_21 = np.array([1])

    wasserstein_convergence(*create_data(10,
                                         [500 * 2**a for a in range(7)],
                                         G_21,
                                         1,
                                         ground_truth_21,
                                         "stuart_21"),
                            "stuart_21")

    print("Stuart examle 22")
    ground_truth_22 = np.array([1])

    wasserstein_convergence(*create_data(10,
                                         [500 * 2**a for a in range(7)],
                                         G_22,
                                         1,
                                         ground_truth_22,
                                         "stuart_22"),
                            "stuart_22")


if __name__ == '__main__':
    main()
