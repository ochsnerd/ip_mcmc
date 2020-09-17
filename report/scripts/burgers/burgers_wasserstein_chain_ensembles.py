import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.stats import norm

from ip_mcmc import (EnsembleManager,
                     MCMCSampler,
                     pCNAccepter,
                     StandardRWAccepter,
                     ConstrainAccepter,
                     CountedAccepter,
                     ConstSteppCNProposer,
                     EvolutionPotential,
                     GaussianDistribution,
                     ConstStepStandardRWProposer,
                     VarStepStandardRWProposer)

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/burgers")
from utilities import (FVMObservationOperator,
                       PerturbedRiemannIC,
                       RusanovMCMC,
                       Measurer,
                       store_figure,
                       load_or_compute,
                       BurgersEquation,
                       autocorrelation,
                       wasserstein_distance,
                       DATA_DIR)


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


def is_valid_IC(u):
    """Return True if u is a valid initial condition.

    Namely it returns False if the initial shock location is
    outside of the domain.
    """
    s = u[2] + Settings.Prior.mean[2]
    return (s > Settings.Simulation.domain[0] and
            s < Settings.Simulation.domain[1])


class Settings:
    """'Static' class to collect the settings for a MCMC simulation"""
    # 'Attributes' that derive from other attributes need to be impleneted
    # using a getter-method, so that they get updated when the thing
    # they depend on changes.
    class Simulation:
        class IC:
            names = ["delta_1", "delta_2", "sigma_0"]
            delta_1 = 0.025
            delta_2 = -0.025
            sigma = -0.02
            ground_truth = np.array([delta_1, delta_2, sigma])

        domain = (-1, 1)
        N_gridpoints = 128
        T_end = 1
        flux = BurgersEquation.flux
        flux_prime = BurgersEquation.flux_prime

        @staticmethod
        def get_xvals():
            return RusanovMCMC(None, None,
                               Settings.Simulation.domain,
                               Settings.Simulation.N_gridpoints,
                               0).FVM.x[1:-1]

    class Measurement:
        points = [-0.5, -0.25, 0.25, 0.5, 0.75]
        interval = 0.1

    class Noise:
        mean = np.array([0] * 5)
        std_dev = 0.05
        covariance = std_dev**2 * np.identity(5)

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Noise.mean,
                                        Settings.Noise.covariance)

    class Prior:
        mean = np.array([1.5,    # delta_1
                         0.25,   # delta_2
                         -0.5])  # sigma
        std_dev = 0.25
        covariance = std_dev**2 * np.identity(len(mean))

        @staticmethod
        def get_distribution():
            return GaussianDistribution(Settings.Prior.mean,
                                        Settings.Prior.covariance)

    class Sampling:
        step = PWLinear(0.05, 0.001, 250)
        u_0 = np.zeros(3)
        N = 5000
        burn_in = 250
        sample_interval = 20
        rng = np.random.default_rng(2)

    @staticmethod
    def filename():
        return f"burgers_RW_n={Settings.Sampling.N}_b={Settings.Sampling.step}"


def create_integrator():
    return RusanovMCMC(Settings.Simulation.flux,
                       Settings.Simulation.flux_prime,
                       Settings.Simulation.domain,
                       Settings.Simulation.N_gridpoints,
                       Settings.Simulation.T_end)


def create_measurer():
    return Measurer(Settings.Measurement.points,
                    Settings.Measurement.interval,
                    Settings.Simulation.get_xvals())


def create_mcmc_sampler():
    # Proposer
    prior = Settings.Prior.get_distribution()
    # proposer = ConstSteppCNProposer(Settings.Sampling.step, prior)
    # proposer = ConstStepStandardRWProposer(Settings.Sampling.step, prior)
    proposer = VarStepStandardRWProposer(Settings.Sampling.step, prior)

    # Accepter
    integrator = create_integrator()
    measurer = create_measurer()
    IC_true = PerturbedRiemannIC(Settings.Simulation.IC.ground_truth)
    observation_operator = FVMObservationOperator(PerturbedRiemannIC,
                                                  Settings.Prior.mean,
                                                  integrator,
                                                  measurer)

    ground_truth = measurer(integrator(IC_true))
    noise = Settings.Noise.get_distribution()
    potential = EvolutionPotential(observation_operator,
                                   ground_truth,
                                   noise)
    # accepter = CountedAccepter(pCNAccepter(potential))
    accepter = ConstrainAccepter(CountedAccepter(StandardRWAccepter(potential, prior)), is_valid_IC)

    return MCMCSampler(proposer, accepter)


def create_data(ensemble_size, chain_lengths, ref_length):
    rngs = [np.random.default_rng(i) for i in range(ensemble_size)]

    # Compute chains
    sampler = create_mcmc_sampler()

    ensembles = []
    for n_samples in chain_lengths:
        Settings.Sampling.N = n_samples
        chain_start = partial(sampler.run,
                              Settings.Sampling.u_0,
                              Settings.Sampling.N,
                              Settings.Sampling.burn_in,
                              Settings.Sampling.sample_interval)

        ensemble_manager = EnsembleManager(DATA_DIR,
                                           f"{Settings.filename()}_"
                                           f"E={ensemble_size}")

        ensemble = ensemble_manager.compute(chain_start,
                                            rngs,
                                            ensemble_size)

        ensembles.append(ensemble)

    # Compute reference
    ref_chain_start = partial(sampler.run,
                              Settings.Sampling.u_0,
                              ref_length,
                              Settings.Sampling.burn_in,
                              Settings.Sampling.sample_interval)

    ref_manager = EnsembleManager(DATA_DIR,
                                  f"{Settings.filename()}_"
                                  f"ref_N={ref_length}")

    ref_chain = ref_manager.compute(ref_chain_start, rngs[:1], 1)[0, :, :]

    # Add prior mean
    for ensemble, chain_length in zip(ensembles, chain_lengths):
        for j in range(ensemble_size):
            for i in range(chain_length):
                ensemble[j, :, i] += Settings.Prior.mean

    for i in range(ref_length):
        ref_chain[:, i] += Settings.Prior.mean

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


def show_ensemble(ensemble):
    """Plz write docstring"""
    for k in range(ensemble.shape[0] - 1):
        for i in range(3):
            plt.plot(ensemble[k, i, :])

    for i in range(3):
        plt.plot(ensemble[-1, i, :], label=Settings.Simulation.IC.names[i])

    for p in Settings.Measurement.points:
        l = p - Settings.Measurement.interval
        r = p + Settings.Measurement.interval
        plt.axhspan(l, r, facecolor='r', alpha=0.3)

    shock_locs = np.zeros_like(ensemble[0, 0, :])
    rarefactions = []
    for i in range(len(shock_locs)):
        try:
            shock_locs[i] = BurgersEquation.riemann_shock_pos(ensemble[0, 0, i] + 1,
                                                              ensemble[0, 1, i],
                                                              ensemble[0, 2, i],
                                                              1)
        except AssertionError:
            rarefactions += [i]
    plt.plot(shock_locs, color='r')
    if rarefactions:
        print(f"{len(rarefactions)} rarefactions during sampling")
        for i in rarefactions:
            plt.axvline(i, color='r', alpha=0.05)

    plt.legend()
    plt.show()

    # densities
    intervals = [(-2, 2)] * 3
    priors = [norm(loc=mu, scale=Settings.Prior.std_dev)
              for mu in Settings.Prior.mean]
    fig, plts = plt.subplots(1, 3, figsize=(20, 10))

    plot_info = list(zip(intervals,
                         (0.025, -0.025, -0.02),
                         Settings.Simulation.IC.names,
                         priors,
                         plts))

    for k in range(ensemble.shape[0] - 1):
        for i, (interval, _, __, ___, ax) in enumerate(plot_info):
            x_range = np.linspace(*interval, num=500)
            ax.hist(ensemble[k, i, :], density=True, alpha=0.5)

    for i, (interval, true_val, name, prior, ax) in enumerate(plot_info):
        x_range = np.linspace(*interval, num=500)
        ax.plot(x_range, [prior.pdf(x) for x in x_range])
        ax.hist(ensemble[-1, i, :], density=True, color='b', alpha=0.5)
        ax.axvline(true_val, c='r')
        ax.set_title(f"Posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    plt.show()


def main():
    ensemble_size = 10
    chain_lengths = [250, 500, 1000, 2000]
    ref_length = 5000
    ensembles, ref_chain = create_data(ensemble_size,
                                       chain_lengths,
                                       ref_length)

    for i, name in enumerate(Settings.Simulation.IC.names):
        wasserstein_convergence([ensemble[:, i, :].reshape(ensemble_size,
                                                           1,
                                                           chain_length)
                                 for ensemble, chain_length in zip(ensembles, chain_lengths)],
                                ref_chain[i, :].reshape(1, ref_length),
                                f"{Settings.filename()}_{name}")

    for ensemble in ensembles:
        show_ensemble(ensemble)

    show_ensemble(np.expand_dims(ref_chain, 0))


if __name__ == '__main__':
    main()
