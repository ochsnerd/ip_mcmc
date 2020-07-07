import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
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
                       wasserstein_distance)


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

    return MCMCSampler(proposer, accepter, Settings.Sampling.rng)


def main():
    chain_length = 100000
    data = create_data(chain_length)

    # support of the data
    data_ndims = data[0].shape[0]
    assert data_ndims == 3, ""
    intervals = np.empty((data_ndims, 2))
    for i in range(data_ndims):
        intervals[i, :] = [min(chain[i, :].min() for chain in data),
                           max(chain[i, :].max() for chain in data)]

    # bin data
    n_bins = 20

    # data_binned = [np.histogram(chain[0, :], bins=n_bins, range=intervals[0])[0]
    #                for chain in data]
    # data_binned = [binned_chain / np.sum(binned_chain) for binned_chain in data_binned]

    # ground_truth, _ = np.histogram([Settings.Simulation.IC.delta_1], bins=n_bins, range=intervals[0])
    data_binned = [np.histogramdd(chain.T, bins=n_bins, range=intervals)[0]
                   for chain in data]
    data_binned = [binned_chain / np.sum(binned_chain) for binned_chain in data_binned]

    ground_truth, _ = np.histogramdd(Settings.Simulation.IC.ground_truth.reshape(1,3), bins=n_bins, range=intervals)
    # -----------------------
    # data_binned = [np.histogramdd(chain.T, bins=n_bins, range=intervals, density=False)[0]
    #                for chain in data]


    # data_binned = [binned_chain / np.sum(binned_chain) for binned_chain in data_binned]

    # ground_truth, _ = np.histogramdd(Settings.Simulation.IC.ground_truth.reshape(1,3),
    #                                  bins=n_bins, range=intervals, density=False)

    # _, (bd1, bd2, bs0) = np.histogramdd(data[0].T, bins=n_bins, range=intervals, density=False)

    # # print(np.sum(H, axis=(1,2)))
    # for i, H in enumerate(data_binned):
    #     plt.plot(bd1[:-1], np.sum(H, axis=(1,2)), label=i)

    # plt.plot(bd1[:-1], np.sum(ground_truth, axis=(1,2)), label="gt")

    # plt.legend()
    # plt.show()

    # return
    # # plt.plot(bd1[:-1], np.sum(ground_truth, axis=(1,2)))
    # # plt.show()

    # H, _ = np.histogram(data[-1][0, :], bins=n_bins, range=intervals[0], density=True)

    # plt.plot(_[:-1], H)
    # plt.show()

    # return

    # print(np.sum(data[-1], axis=(0,1)))
    # plt.plot(np.sum(data[-1], axis=(0,1)))
    # plt.show()

    # return

    # # generate ground truth
    # ground_truth, _ = np.histogramdd(Settings.Simulation.IC.ground_truth.reshape(1,3),
    #                                  bins=n_bins, range=intervals, density=True)

    # errors = [wasserstein_distance(density, ground_truth, intervals[:1].reshape(1,2)) for density in data_binned]
    errors = [wasserstein_distance(density, ground_truth, intervals) for density in data_binned]

    plt.plot([chain_length / (2**(i+1)) for i in range(len(data))], errors)
    plt.title("$W_1$ for different chain lengths")
    plt.xlabel("Length of the chain")
    plt.ylabel("$W_1$")
    store_figure(Settings.filename() + "_wasserstein_convergence_chain")


def create_data(chain_length):
    # run chain
    Settings.Sampling.N = chain_length

    sampler = create_mcmc_sampler()

    samples_full = load_or_compute(Settings.filename(),
                                   sampler.run,
                                   (Settings.Sampling.u_0,
                                    Settings.Sampling.N,
                                    0,
                                    1))

    samples_full = samples_full.T
    samples = samples_full[:, ::Settings.Sampling.sample_interval]

    # Add pertubations to means
    for i in range(len(samples[0, :])):
        samples[:, i] += Settings.Prior.mean

    chains = []
    for _ in range(4):
        l = int(len(samples[0, :]) / 2)
        chains += [samples[:, l + 1:]]

        samples = samples[:, :l]

    return chains


if __name__ == '__main__':
    main()
